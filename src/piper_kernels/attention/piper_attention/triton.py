"""Pure-Triton backend for Piper Attention.

The kernel keeps Sage-style INT8 QK and FP32 online softmax, but replaces the
FP8 PV path with signed-INT8 V and UINT8-by-INT8 tensor-core dots. The generic
path folds one signed-INT8 scale per V row into the probability operand. The
long SM89 D128 specialization shares a centered scale across each 64-key tile
and uses a split FP16 numerator recurrence. Native mixed-sign MMA is used on
supported NVIDIA targets; an algebraically identical affine signed-INT8
formulation is retained as the portable control.
"""

# Triton's JIT launcher accepts compile-time options not represented in its
# Python call signature.
# pyright: reportCallIssue=false

from dataclasses import dataclass
from typing import Any, cast

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from piper_kernels._triton.mixed_int8 import (
    install_uint8_int8_dot_hook,
    uint8_int8_dot,
)
from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.kernels.qk_quantization.int8.sage import (
    triton as qk_quantization,
)
from piper_kernels.attention.scheduling import select_query_block

_BLOCK_N = 64
_MEAN_CHUNK_N = 1024
_MEAN_BLOCK_N = 64
_MEAN_BLOCK_D = 64
_P_UINT8_RANGE = tl.constexpr(255.0)
_P_UINT8_LOG2_RANGE = tl.constexpr(7.994353436858858)
_P_ZERO_POINT = tl.constexpr(128)
_V_INT8_RANGE = tl.constexpr(127.0)
_SCALE_EPSILON = tl.constexpr(1e-7)
_LOG2_E = tl.constexpr(1.4426950408889634)


@triton.jit
def _ptx_float32_to_uint8x4(values):
    """Truncate and saturate four probability codes with packed SM72+ PTX."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .s32 a, b, c, d;
            .reg .b32 lo;
            cvt.rzi.s32.f32 a, $1;
            cvt.rzi.s32.f32 b, $2;
            cvt.rzi.s32.f32 c, $3;
            cvt.rzi.s32.f32 d, $4;
            cvt.pack.sat.u8.s32.b32 lo, d, c, 0;
            cvt.pack.sat.u8.s32.b32 $0, b, a, lo;
        }
        """,
        constraints="=r,f,f,f,f",
        args=[values],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )


@triton.jit
def _kv_mean_partial_kernel(
    key_ptr,
    value_ptr,
    key_partial_ptr,
    value_partial_ptr,
    key_length,
    num_chunks,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    center_value: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_n: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Reduce one raw K/V sequence chunk into compact FP32 partials."""
    chunk = tl.program_id(0)
    feature_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    offsets_n = tl.arange(0, block_n)
    key_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    if center_value:
        value_accumulator = tl.zeros((block_d,), dtype=tl.float32)
    chunk_start = chunk * chunk_n
    for offset in tl.range(0, chunk_n, block_n, disable_licm=True):
        current_n = chunk_start + offset + offsets_n
        mask = (current_n[:, None] < key_length) & (offsets_d[None, :] < head_dim)
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + current_n[:, None] * stride_kn
            + offsets_d[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        key_accumulator += tl.sum(key, axis=0)
        if center_value:
            value = tl.load(
                value_ptr
                + batch * stride_vb
                + head * stride_vh
                + current_n[:, None] * stride_vn
                + offsets_d[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            value_accumulator += tl.sum(value, axis=0)
    output_offsets = (batch_head * num_chunks + chunk) * head_dim + offsets_d
    tl.store(key_partial_ptr + output_offsets, key_accumulator, mask=offsets_d < head_dim)
    if center_value:
        tl.store(
            value_partial_ptr + output_offsets,
            value_accumulator,
            mask=offsets_d < head_dim,
        )


@triton.jit
def _kv_mean_finalize_kernel(
    key_partial_ptr,
    value_partial_ptr,
    key_mean_ptr,
    value_mean_ptr,
    key_length,
    num_chunks,
    center_value: tl.constexpr,
    head_dim: tl.constexpr,
    block_chunks: tl.constexpr,
    block_d: tl.constexpr,
):
    """Merge K/V chunk partials into compact FP32 means."""
    batch_head = tl.program_id(0)
    feature_block = tl.program_id(1)
    offsets_c = tl.arange(0, block_chunks)
    offsets_d = feature_block * block_d + tl.arange(0, block_d)
    mask = (offsets_c[:, None] < num_chunks) & (offsets_d[None, :] < head_dim)
    partial_offsets = (
        (batch_head * num_chunks + offsets_c[:, None]) * head_dim
        + offsets_d[None, :]
    )
    key_partials = tl.load(key_partial_ptr + partial_offsets, mask=mask, other=0.0)
    output_offsets = batch_head * head_dim + offsets_d
    output_mask = offsets_d < head_dim
    tl.store(
        key_mean_ptr + output_offsets,
        tl.sum(key_partials, axis=0) / key_length,
        mask=output_mask,
    )
    if center_value:
        value_partials = tl.load(
            value_partial_ptr + partial_offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            value_mean_ptr + output_offsets,
            tl.sum(value_partials, axis=0) / key_length,
            mask=output_mask,
        )


@triton.jit
def _centered_value_row_range_kernel(
    value_ptr,
    value_mean_ptr,
    range_ptr,
    key_length,
    stride_vb,
    stride_vh,
    stride_vn,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Compute centered per-row V ranges without materializing centered V."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + offsets_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    value_mean = tl.load(value_mean_ptr + (batch * heads + head) * head_dim + offsets_d)
    centered = tl.where(valid[:, None], value - value_mean[None, :], 0.0)
    tl.store(
        range_ptr + (batch * heads + head) * key_length + offsets_n,
        tl.max(tl.abs(centered), axis=1),
        mask=valid,
    )


@triton.jit
def _quantize_ordered_key_per_block_kernel(
    key_ptr,
    key_mean_ptr,
    order_ptr,
    output_ptr,
    scale_ptr,
    key_length,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_ob,
    stride_oh,
    stride_on,
    stride_sb,
    stride_sh,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Gather ordered K rows directly into grouped INT8 storage."""
    scale_group = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = scale_group * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    batch_head = batch * heads + head
    source_n = tl.load(
        order_ptr + batch_head * key_length + offsets_n,
        mask=valid,
        other=0,
    )
    mean = tl.load(key_mean_ptr + batch_head * head_dim + offsets_d)
    values = tl.load(
        key_ptr
        + batch * stride_kb
        + head * stride_kh
        + source_n[:, None] * stride_kn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    values = tl.where(valid[:, None], values - mean[None, :], 0.0)
    maximum = tl.max(tl.max(tl.abs(values), axis=1), axis=0)
    scale = maximum / 127.0 + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(values / scale)
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_n[:, None] * stride_on
        + offsets_d[None, :],
        quantized,
        mask=valid[:, None],
    )
    tl.store(scale_ptr + batch * stride_sb + head * stride_sh + scale_group, scale)


@triton.jit
def _quantize_value_per_key_kernel(
    value_ptr,
    key_ptr,
    query_ptr,
    value_mean_ptr,
    key_mean_ptr,
    order_ptr,
    scale_multiplier_ptr,
    log_scale_ptr,
    correction_ptr,
    output_ptr,
    key_output_ptr,
    key_scale_ptr,
    query_output_ptr,
    query_scale_ptr,
    key_length,
    storage_key_length,
    softmax_scale,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_qb,
    stride_qh,
    stride_qn,
    stride_ob,
    stride_oh,
    stride_od,
    stride_ok,
    center_value: tl.constexpr,
    ordered: tl.constexpr,
    store_correction: tl.constexpr,
    block_value_scale: tl.constexpr,
    fuse_key_quantization: tl.constexpr,
    fuse_query_quantization: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
):
    """Quantize centered V with one symmetric signed-INT8 scale per key."""
    key_block = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    offsets_n = key_block * block_n + tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    valid = offsets_n < key_length
    batch_head = batch * heads + head
    if fuse_query_quantization:
        query = tl.load(
            query_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_n[:, None] * stride_qn
            + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        query_quantization_scale = (
            tl.max(tl.abs(query), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
        )
        query_quantized = qk_quantization.round_to_int8(
            query / query_quantization_scale[:, None]
        )
        tl.store(
            query_output_ptr
            + (batch_head * key_length + offsets_n[:, None]) * head_dim
            + offsets_d[None, :],
            query_quantized,
            mask=valid[:, None],
        )
        tl.store(
            query_scale_ptr + batch_head * key_length + offsets_n,
            query_quantization_scale * (softmax_scale * _LOG2_E),
            mask=valid,
        )
    if ordered:
        source_n = tl.load(
            order_ptr + batch_head * key_length + offsets_n,
            mask=valid,
            other=0,
        )
    else:
        source_n = offsets_n
    if fuse_key_quantization:
        key = tl.load(
            key_ptr
            + batch * stride_kb
            + head * stride_kh
            + source_n[:, None] * stride_kn
            + offsets_d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        key_mean = tl.load(key_mean_ptr + batch_head * head_dim + offsets_d)
        key = tl.where(valid[:, None], key - key_mean[None, :], 0.0)
        key_quantization_scale = (
            tl.max(tl.abs(key), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
        )
        key_quantized = qk_quantization.round_to_int8(
            key / key_quantization_scale[:, None]
        )
        tl.store(
            key_output_ptr
            + (batch_head * storage_key_length + offsets_n[:, None]) * head_dim
            + offsets_d[None, :],
            key_quantized,
            mask=valid[:, None],
        )
        tl.store(
            key_scale_ptr + batch_head * key_length + offsets_n,
            key_quantization_scale,
            mask=valid,
        )
    value = tl.load(
        value_ptr
        + batch * stride_vb
        + head * stride_vh
        + source_n[:, None] * stride_vn
        + offsets_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    if center_value:
        value_mean = tl.load(value_mean_ptr + batch_head * head_dim + offsets_d)
        value = tl.where(valid[:, None], value - value_mean[None, :], 0.0)
    scale = tl.max(tl.abs(value), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
    quantized = qk_quantization.round_to_int8(value / scale[:, None])
    if block_value_scale:
        maximum_scale = tl.max(scale, axis=0)
        quantized = qk_quantization.round_to_int8(value / maximum_scale)
        tl.store(
            log_scale_ptr
            + batch_head * tl.cdiv(key_length, block_n)
            + key_block,
            maximum_scale,
        )
    else:
        tl.store(
            scale_multiplier_ptr + batch_head * key_length + offsets_n,
            scale * _P_UINT8_RANGE,
            mask=valid,
        )
        tl.store(
            log_scale_ptr + batch_head * key_length + offsets_n,
            tl.log2(scale),
            mask=valid,
        )
    if store_correction:
        correction_offsets = (
            (batch_head * tl.cdiv(key_length, block_n) + key_block) * head_dim
            + offsets_d
        )
        tl.store(
            correction_ptr + correction_offsets,
            tl.sum(quantized.to(tl.int32), axis=0),
        )
    tl.store(
        output_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_d[None, :] * stride_od
        + offsets_n[:, None] * stride_ok,
        quantized,
        mask=valid[:, None],
    )


@triton.jit
def _load_key_tile(
    key_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    storage_key_length,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return key_ptr.load([batch_head, start_n, 0]).reshape((block_n, head_dim)).T
    else:
        return tl.load(
            key_ptr
            + (batch_head * storage_key_length + current_n[None, :]) * head_dim
            + offsets_d[:, None],
            mask=current_n[None, :] < key_length,
            other=0,
        )


@triton.jit
def _load_value_tile(
    value_ptr,
    batch_head,
    start_n,
    current_n,
    offsets_d,
    key_length,
    storage_key_length,
    feature_start: tl.constexpr,
    feature_block: tl.constexpr,
    head_dim: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
):
    if use_tensor_descriptors:
        return value_ptr.load([batch_head, feature_start, start_n]).reshape(
            (feature_block, block_n)
        ).T
    else:
        return tl.load(
            value_ptr
            + (batch_head * head_dim + feature_start + offsets_d[None, :])
            * storage_key_length
            + current_n[:, None],
            mask=current_n[:, None] < key_length,
            other=0,
        )


@triton.jit
def _piper_sm89_d128_tile(  # noqa: PLR0912, PLR0915 - hot tile stays monolithic
    query,
    query_scale,
    key_ptr,
    value_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_scale_coordinate_ptr,
    accumulator_low,
    accumulator_high,
    denominator,
    running_max,
    batch_head,
    start_n,
    offsets_m,
    offsets_n,
    offsets_d,
    key_length,
    diagonal_or_tail: tl.constexpr,
    block_value_scale: tl.constexpr,
    round_probability_codes: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    block_n: tl.constexpr,
):
    """Advance the branch-minimal SM89 D128 recurrence by one key tile."""
    head_dim: tl.constexpr = 128
    half_head_dim: tl.constexpr = 64
    current_n = start_n + offsets_n
    key = tl.load(
        key_ptr
        + (batch_head * key_length + current_n[None, :]) * head_dim
        + offsets_d[:, None]
    )
    integer_scores = tl.dot(query, key, out_dtype=tl.int32)
    key_scale = tl.load(key_scale_ptr + batch_head * key_length + current_n)
    scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]
    if diagonal_or_tail:
        valid_keys = current_n[None, :] <= offsets_m[:, None]
        scores = tl.where(valid_keys, scores, -float("inf"))

    if block_value_scale:
        block_max = tl.max(scores, axis=1) - _P_UINT8_LOG2_RANGE
    else:
        value_log_scale = tl.load(
            value_scale_coordinate_ptr + batch_head * key_length + current_n
        )
        block_max = tl.max(scores + value_log_scale[None, :], axis=1)
    next_max = tl.maximum(running_max, block_max)
    old_weight = tl.exp2(running_max - next_max)
    if block_value_scale:
        probabilities = tl.exp2(scores - next_max[:, None])
        if diagonal_or_tail:
            probabilities = tl.where(valid_keys, probabilities, 0.0)
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1)
    else:
        current_weight = tl.exp2(block_max - next_max)
        probabilities = tl.exp2(scores - block_max[:, None])
        if diagonal_or_tail:
            probabilities = tl.where(valid_keys, probabilities, 0.0)
        denominator = denominator * old_weight + tl.sum(probabilities, axis=1) * current_weight

    if block_value_scale:
        if use_packed_probability_conversion:
            probability_values = (
                probabilities + 0.5 if round_probability_codes else probabilities
            )
            probability_uint8 = _ptx_float32_to_uint8x4(probability_values)
        elif round_probability_codes:
            probability_uint8 = (probabilities + 0.5).to(tl.uint8)
        else:
            probability_uint8 = probabilities.to(tl.uint8)
    else:
        value_scale_multiplier = tl.load(
            value_scale_multiplier_ptr + batch_head * key_length + current_n
        )
        probability_codes = tl.minimum(
            _P_UINT8_RANGE,
            probabilities * value_scale_multiplier[None, :] + 0.5,
        ).to(tl.int32)
        probability_uint8 = probability_codes.to(tl.uint8)
    offsets_vd = tl.arange(0, half_head_dim)
    value_base = batch_head * head_dim * key_length + current_n[:, None]
    value_low = tl.load(
        value_ptr + value_base + offsets_vd[None, :] * key_length
    )
    value_high = tl.load(
        value_ptr
        + value_base
        + (half_head_dim + offsets_vd[None, :]) * key_length
    )
    partial_low = uint8_int8_dot(probability_uint8, value_low)
    partial_high = uint8_int8_dot(probability_uint8, value_high)
    if block_value_scale:
        coordinate_block = batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        block_scale = tl.load(value_scale_coordinate_ptr + coordinate_block)
        partial_low_scaled = (
            partial_low.to(tl.float32) * (block_scale / 65536.0)
        ).to(tl.float16)
        partial_high_scaled = (
            partial_high.to(tl.float32) * (block_scale / 65536.0)
        ).to(tl.float16)
        accumulator_low = accumulator_low * old_weight[:, None].to(tl.float16) + partial_low_scaled
        accumulator_high = (
            accumulator_high * old_weight[:, None].to(tl.float16) + partial_high_scaled
        )
    else:
        partial_low_scaled = (partial_low.to(tl.float32) * (1.0 / 65536.0)).to(
            tl.float16
        )
        partial_high_scaled = (partial_high.to(tl.float32) * (1.0 / 65536.0)).to(
            tl.float16
        )
        current_weight_fp16 = current_weight[:, None].to(tl.float16)
        accumulator_low = (
            accumulator_low * old_weight[:, None].to(tl.float16)
            + partial_low_scaled * current_weight_fp16
        )
        accumulator_high = (
            accumulator_high * old_weight[:, None].to(tl.float16)
            + partial_high_scaled * current_weight_fp16
        )
    return accumulator_low, accumulator_high, denominator, next_max


@triton.jit
def _piper_attention_sm89_d128_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_scale_coordinate_ptr,
    value_mean_ptr,
    output_ptr,
    query_length,
    key_length,
    softmax_scale,
    is_causal: tl.constexpr,
    reverse_causal_blocks: tl.constexpr,
    block_value_scale: tl.constexpr,
    round_probability_codes: tl.constexpr,
    use_packed_probability_conversion: tl.constexpr,
    center_value: tl.constexpr,
    fuse_query_quantization: tl.constexpr,
    loop_num_stages: tl.constexpr,
    disable_loop_licm: tl.constexpr,
    heads: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """Branch-minimal centered native-UINT8 SM89 D128 specialization."""
    head_dim: tl.constexpr = 128
    half_head_dim: tl.constexpr = 64
    query_block = tl.program_id(0)
    if is_causal and reverse_causal_blocks:
        query_block = tl.num_programs(0) - 1 - query_block
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    if fuse_query_quantization:
        query_values = tl.load(
            query_ptr
            + (batch_head * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :]
        ).to(tl.float32)
        query_quantization_scale = (
            tl.max(tl.abs(query_values), axis=1) / _V_INT8_RANGE + _SCALE_EPSILON
        )
        query = qk_quantization.round_to_int8(
            query_values / query_quantization_scale[:, None]
        )
        query_scale = query_quantization_scale * (softmax_scale * _LOG2_E)
    else:
        query = tl.load(
            query_ptr
            + (batch_head * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :]
        )
        query_scale = tl.load(query_scale_ptr + batch_head * query_length + offsets_m)
    accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
    accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)

    if is_causal:
        prefix_end = query_block * block_m
        for start_n in tl.range(
            0,
            prefix_end,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=disable_loop_licm,
        ):
            accumulator_low, accumulator_high, denominator, running_max = (
                _piper_sm89_d128_tile(
                    query,
                    query_scale,
                    key_ptr,
                    value_ptr,
                    key_scale_ptr,
                    value_scale_multiplier_ptr,
                    value_scale_coordinate_ptr,
                    accumulator_low,
                    accumulator_high,
                    denominator,
                    running_max,
                    batch_head,
                    start_n,
                    offsets_m,
                    offsets_n,
                    offsets_d,
                    key_length,
                    diagonal_or_tail=False,
                    block_value_scale=block_value_scale,
                    round_probability_codes=round_probability_codes,
                    use_packed_probability_conversion=use_packed_probability_conversion,
                    block_n=block_n,
                )
            )
        end_n = (query_block + 1) * block_m
        for start_n in tl.range(
            prefix_end,
            end_n,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=disable_loop_licm,
        ):
            accumulator_low, accumulator_high, denominator, running_max = (
                _piper_sm89_d128_tile(
                    query,
                    query_scale,
                    key_ptr,
                    value_ptr,
                    key_scale_ptr,
                    value_scale_multiplier_ptr,
                    value_scale_coordinate_ptr,
                    accumulator_low,
                    accumulator_high,
                    denominator,
                    running_max,
                    batch_head,
                    start_n,
                    offsets_m,
                    offsets_n,
                    offsets_d,
                    key_length,
                    diagonal_or_tail=True,
                    block_value_scale=block_value_scale,
                    round_probability_codes=round_probability_codes,
                    use_packed_probability_conversion=use_packed_probability_conversion,
                    block_n=block_n,
                )
            )
    else:
        for start_n in tl.range(
            0,
            key_length,
            block_n,
            num_stages=loop_num_stages,
            disable_licm=disable_loop_licm,
        ):
            accumulator_low, accumulator_high, denominator, running_max = (
                _piper_sm89_d128_tile(
                    query,
                    query_scale,
                    key_ptr,
                    value_ptr,
                    key_scale_ptr,
                    value_scale_multiplier_ptr,
                    value_scale_coordinate_ptr,
                    accumulator_low,
                    accumulator_high,
                    denominator,
                    running_max,
                    batch_head,
                    start_n,
                    offsets_m,
                    offsets_n,
                    offsets_d,
                    key_length,
                    diagonal_or_tail=False,
                    block_value_scale=block_value_scale,
                    round_probability_codes=round_probability_codes,
                    use_packed_probability_conversion=use_packed_probability_conversion,
                    block_n=block_n,
                )
            )

    denominator_code_scale: tl.constexpr = (
        1.0 / 65536.0 if block_value_scale else _P_UINT8_RANGE / 65536.0
    )
    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    output_low = accumulator_low.to(tl.float32) / (
        denominator_safe * denominator_code_scale
    )
    output_high = accumulator_high.to(tl.float32) / (
        denominator_safe * denominator_code_scale
    )
    offsets_vd = tl.arange(0, half_head_dim)
    if center_value:
        value_mean_base = value_mean_ptr + batch_head * head_dim
        output_low += tl.load(value_mean_base + offsets_vd)[None, :]
        output_high += tl.load(value_mean_base + half_head_dim + offsets_vd)[None, :]
    output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
    tl.store(output_base + offsets_vd[None, :], output_low)
    tl.store(output_base + half_head_dim + offsets_vd[None, :], output_high)


@triton.jit
def _piper_attention_kernel(  # noqa: PLR0912, PLR0915
    query_ptr,
    key_ptr,
    value_ptr,
    query_scale_ptr,
    key_scale_ptr,
    value_scale_multiplier_ptr,
    value_log_scale_ptr,
    value_correction_ptr,
    value_mean_ptr,
    output_ptr,
    query_length,
    key_length,
    storage_key_length,
    query_block_offset: tl.constexpr,
    is_causal: tl.constexpr,
    grouped_qk: tl.constexpr,
    native_uint8: tl.constexpr,
    split_pv_head_dim: tl.constexpr,
    scaled_fp16_numerator: tl.constexpr,
    center_value: tl.constexpr,
    unmasked_query_tiles: tl.constexpr,
    unmasked_self_attention: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    reverse_causal_blocks: tl.constexpr,
    loop_num_stages: tl.constexpr,
    disable_loop_licm: tl.constexpr,
    block_value_scale: tl.constexpr,
    round_probability_codes: tl.constexpr,
):
    """Fused exact-log UINT8-P/INT8-V online attention."""
    query_block = tl.program_id(0)
    if is_causal and reverse_causal_blocks:
        query_block = tl.num_programs(0) - 1 - query_block
    query_block += query_block_offset
    head = tl.program_id(1)
    batch = tl.program_id(2)
    batch_head = batch * heads + head
    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    if unmasked_query_tiles:
        valid_queries = tl.full((block_m,), True, dtype=tl.int1)
    else:
        valid_queries = offsets_m < query_length

    query = tl.load(
        query_ptr
        + ((batch_head * query_length + offsets_m[:, None]) * head_dim)
        + offsets_d[None, :],
        mask=valid_queries[:, None],
        other=0,
    )
    if grouped_qk:
        query_scale = tl.load(
            query_scale_ptr
            + batch_head * tl.cdiv(query_length, 32)
            + offsets_m // 32,
            mask=valid_queries,
            other=0.0,
        )
    else:
        query_scale = tl.load(
            query_scale_ptr + batch_head * query_length + offsets_m,
            mask=valid_queries,
            other=0.0,
        )

    if split_pv_head_dim:
        half_head_dim: tl.constexpr = head_dim // 2
        offsets_vd = tl.arange(0, half_head_dim)
        if scaled_fp16_numerator:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float16)
        else:
            accumulator_low = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
            accumulator_high = tl.zeros((block_m, half_head_dim), dtype=tl.float32)
    elif scaled_fp16_numerator:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float16)
    else:
        accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    running_max = tl.full((block_m,), -float("inf"), dtype=tl.float32)
    end_n = key_length
    if is_causal:
        end_n = tl.minimum(key_length, (query_block + 1) * block_m)
        full_key_end = key_length // block_n * block_n
        causal_prefix_end = query_block * block_m // block_n * block_n
        causal_prefix_end = tl.minimum(causal_prefix_end, full_key_end)

    for start_n in tl.range(
        0,
        end_n,
        block_n,
        num_stages=loop_num_stages,
        disable_licm=disable_loop_licm,
    ):
        current_n = start_n + offsets_n
        key = _load_key_tile(
            key_ptr,
            batch_head,
            start_n,
            current_n,
            offsets_d,
            key_length,
            storage_key_length,
            head_dim,
            block_n,
            use_tensor_descriptors,
        )
        integer_scores = tl.dot(query, key, out_dtype=tl.int32)
        if grouped_qk:
            key_scale = tl.load(
                key_scale_ptr
                + batch_head * tl.cdiv(key_length, block_n)
                + start_n // block_n
            )
            scores = integer_scores.to(tl.float32) * (query_scale * key_scale)[:, None]
        else:
            key_scale = tl.load(
                key_scale_ptr + batch_head * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            scores = integer_scores.to(tl.float32) * query_scale[:, None] * key_scale[None, :]

        if is_causal:
            if start_n < causal_prefix_end:
                valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
            else:
                valid_keys = (current_n[None, :] < key_length) & (
                    current_n[None, :] <= offsets_m[:, None]
                )
                scores = tl.where(
                    valid_queries[:, None] & valid_keys,
                    scores,
                    -float("inf"),
                )
        elif unmasked_self_attention:
            valid_keys = tl.full((block_m, block_n), True, dtype=tl.int1)
        else:
            valid_keys = current_n[None, :] < key_length
            scores = tl.where(valid_queries[:, None] & valid_keys, scores, -float("inf"))

        if block_value_scale:
            block_max = tl.max(scores, axis=1) - _P_UINT8_LOG2_RANGE
        else:
            value_log_scale = tl.load(
                value_log_scale_ptr + batch_head * key_length + current_n,
                mask=current_n < key_length,
                other=0.0,
            )
            shifted_scores = scores + value_log_scale[None, :]
            block_max = tl.max(shifted_scores, axis=1)
        next_max = tl.maximum(running_max, block_max)
        old_weight = tl.where(
            valid_queries,
            tl.exp2(running_max - next_max),
            0.0,
        )
        if block_value_scale:
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - next_max[:, None]),
                0.0,
            )
            denominator = denominator * old_weight + tl.sum(probabilities, axis=1)
            current_weight = tl.full((block_m,), 1.0, dtype=tl.float32)
        else:
            current_weight = tl.where(
                valid_queries,
                tl.exp2(block_max - next_max),
                0.0,
            )
            probabilities = tl.where(
                valid_queries[:, None] & valid_keys,
                tl.exp2(scores - block_max[:, None]),
                0.0,
            )
            denominator = (
                denominator * old_weight
                + tl.sum(probabilities, axis=1) * current_weight
            )
        if block_value_scale and native_uint8:
            if round_probability_codes:
                probability_uint8 = (probabilities + 0.5).to(tl.uint8)
            else:
                probability_uint8 = probabilities.to(tl.uint8)
        else:
            if block_value_scale:
                probability_codes = probabilities.to(tl.int32)
            else:
                value_scale_multiplier = tl.load(
                    value_scale_multiplier_ptr + batch_head * key_length + current_n,
                    mask=current_n < key_length,
                    other=0.0,
                )
                probability_codes = tl.minimum(
                    _P_UINT8_RANGE,
                    probabilities * value_scale_multiplier[None, :] + 0.5,
                ).to(tl.int32)
            if native_uint8:
                probability_uint8 = probability_codes.to(tl.uint8)
            else:
                probability_int8 = (probability_codes - _P_ZERO_POINT).to(tl.int8)
        correction_block = batch_head * tl.cdiv(key_length, block_n) + start_n // block_n
        if block_value_scale:
            block_scale = tl.load(value_log_scale_ptr + correction_block)

        if split_pv_head_dim:
            value_low = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_vd,
                key_length,
                storage_key_length,
                0,
                half_head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            value_high = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_vd,
                key_length,
                storage_key_length,
                half_head_dim,
                half_head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            if native_uint8:
                partial_low = uint8_int8_dot(probability_uint8, value_low)
                partial_high = uint8_int8_dot(probability_uint8, value_high)
            else:
                correction_base = correction_block * head_dim
                correction_low = (
                    tl.load(value_correction_ptr + correction_base + offsets_vd).to(tl.int32)
                    << 7
                )
                correction_high = (
                    tl.load(
                        value_correction_ptr
                        + correction_base
                        + half_head_dim
                        + offsets_vd
                    ).to(tl.int32)
                    << 7
                )
                partial_low = tl.dot(
                    probability_int8,
                    value_low,
                    acc=tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                    + correction_low[None, :],
                    out_dtype=tl.int32,
                )
                partial_high = tl.dot(
                    probability_int8,
                    value_high,
                    acc=tl.zeros((block_m, half_head_dim), dtype=tl.int32)
                    + correction_high[None, :],
                    out_dtype=tl.int32,
                )
            if scaled_fp16_numerator:
                if block_value_scale:
                    partial_low_scaled = (
                        partial_low.to(tl.float32) * (block_scale / 65536.0)
                    ).to(tl.float16)
                    partial_high_scaled = (
                        partial_high.to(tl.float32) * (block_scale / 65536.0)
                    ).to(tl.float16)
                else:
                    partial_low_scaled = (
                        partial_low.to(tl.float32) * (1.0 / 65536.0)
                    ).to(tl.float16)
                    partial_high_scaled = (
                        partial_high.to(tl.float32) * (1.0 / 65536.0)
                    ).to(tl.float16)
                accumulator_low = (
                    accumulator_low * old_weight[:, None].to(tl.float16)
                    + partial_low_scaled * current_weight[:, None].to(tl.float16)
                )
                accumulator_high = (
                    accumulator_high * old_weight[:, None].to(tl.float16)
                    + partial_high_scaled * current_weight[:, None].to(tl.float16)
                )
            else:
                numerator_scale = (
                    block_scale if block_value_scale else (1.0 / _P_UINT8_RANGE)
                )
                accumulator_low = (
                    accumulator_low * old_weight[:, None]
                    + partial_low.to(tl.float32)
                    * numerator_scale
                    * current_weight[:, None]
                )
                accumulator_high = (
                    accumulator_high * old_weight[:, None]
                    + partial_high.to(tl.float32)
                    * numerator_scale
                    * current_weight[:, None]
                )
        else:
            value_tile = _load_value_tile(
                value_ptr,
                batch_head,
                start_n,
                current_n,
                offsets_d,
                key_length,
                storage_key_length,
                0,
                head_dim,
                head_dim,
                block_n,
                use_tensor_descriptors,
            )
            if native_uint8:
                partial = uint8_int8_dot(probability_uint8, value_tile)
            else:
                correction = (
                    tl.load(
                        value_correction_ptr
                        + correction_block * head_dim
                        + offsets_d
                    ).to(tl.int32)
                    << 7
                )
                partial = tl.dot(
                    probability_int8,
                    value_tile,
                    acc=tl.zeros((block_m, head_dim), dtype=tl.int32)
                    + correction[None, :],
                    out_dtype=tl.int32,
                )
            if scaled_fp16_numerator:
                numerator_scale = block_scale if block_value_scale else 1.0
                partial_scaled = (
                    partial.to(tl.float32) * (numerator_scale / 65536.0)
                ).to(tl.float16)
                accumulator = (
                    accumulator * old_weight[:, None].to(tl.float16)
                    + partial_scaled * current_weight[:, None].to(tl.float16)
                )
            else:
                numerator_scale = (
                    block_scale if block_value_scale else (1.0 / _P_UINT8_RANGE)
                )
                accumulator = (
                    accumulator * old_weight[:, None]
                    + partial.to(tl.float32)
                    * numerator_scale
                    * current_weight[:, None]
                )
        running_max = next_max

    denominator_safe = tl.maximum(denominator, 1e-30)[:, None]
    if split_pv_head_dim:
        if scaled_fp16_numerator:
            denominator_code_scale: tl.constexpr = (
                1.0 / 65536.0
                if block_value_scale
                else _P_UINT8_RANGE / 65536.0
            )
            output_low = accumulator_low.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
            output_high = accumulator_high.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
        else:
            output_low = accumulator_low / denominator_safe
            output_high = accumulator_high / denominator_safe
        if center_value:
            value_mean_base = value_mean_ptr + batch_head * head_dim
            output_low += tl.load(value_mean_base + offsets_vd)[None, :]
            output_high += tl.load(
                value_mean_base + half_head_dim + offsets_vd
            )[None, :]
        output_base = output_ptr + (batch_head * query_length + offsets_m[:, None]) * head_dim
        tl.store(
            output_base + offsets_vd[None, :],
            output_low,
            mask=valid_queries[:, None],
        )
        tl.store(
            output_base + half_head_dim + offsets_vd[None, :],
            output_high,
            mask=valid_queries[:, None],
        )
    else:
        if scaled_fp16_numerator:
            denominator_code_scale: tl.constexpr = (
                1.0 / 65536.0
                if block_value_scale
                else _P_UINT8_RANGE / 65536.0
            )
            output = accumulator.to(tl.float32) / (
                denominator_safe * denominator_code_scale
            )
        else:
            output = accumulator / denominator_safe
        if center_value:
            output += tl.load(
                value_mean_ptr + batch_head * head_dim + offsets_d
            )[None, :]
        tl.store(
            output_ptr
            + (batch_head * query_length + offsets_m[:, None]) * head_dim
            + offsets_d[None, :],
            output,
            mask=valid_queries[:, None],
        )


def _compute_kv_means(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    center_value: bool,
    value_mean_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, heads, key_length, head_dim = key.shape
    mean_block_d = 128 if head_dim == 128 else _MEAN_BLOCK_D
    num_chunks = int(triton.cdiv(key_length, _MEAN_CHUNK_N))
    partial_shape = (batch, heads, num_chunks, head_dim)
    key_partial = torch.empty(partial_shape, device=key.device, dtype=torch.float32)
    value_partial = (
        torch.empty_like(key_partial)
        if center_value
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    key_mean = torch.empty((batch, heads, head_dim), device=key.device, dtype=torch.float32)
    value_mean = (
        torch.empty(
            (batch, heads, head_dim),
            device=value.device,
            dtype=value_mean_dtype,
        )
        if center_value
        else torch.empty((1,), device=value.device, dtype=torch.float32)
    )
    _kv_mean_partial_kernel[
        (
            num_chunks,
            int(triton.cdiv(head_dim, mean_block_d)),
            batch * heads,
        )
    ](
        key,
        value,
        key_partial,
        value_partial,
        key_length,
        num_chunks,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        center_value=center_value,
        heads=heads,
        head_dim=head_dim,
        chunk_n=_MEAN_CHUNK_N,
        block_n=_MEAN_BLOCK_N,
        block_d=mean_block_d,
        num_warps=4,
    )
    _kv_mean_finalize_kernel[
        (batch * heads, int(triton.cdiv(head_dim, mean_block_d)))
    ](
        key_partial,
        value_partial,
        key_mean,
        value_mean,
        key_length,
        num_chunks,
        center_value=center_value,
        head_dim=head_dim,
        block_chunks=triton.next_power_of_2(num_chunks),
        block_d=mean_block_d,
        num_warps=4,
    )
    return key_mean, value_mean


def _build_centered_value_order(
    value: torch.Tensor,
    value_mean: torch.Tensor,
) -> torch.Tensor:
    batch, heads, key_length, head_dim = value.shape
    value_range = torch.empty(
        (batch, heads, key_length),
        device=value.device,
        dtype=torch.float32,
    )
    block_n = 32
    _centered_value_row_range_kernel[
        (triton.cdiv(key_length, block_n), heads, batch)
    ](
        value,
        value_mean,
        value_range,
        key_length,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        heads=heads,
        head_dim=head_dim,
        block_n=block_n,
        num_warps=4,
    )
    return torch.argsort(value_range, dim=-1, stable=True).to(torch.int32)


def _prepare_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    key_mean: torch.Tensor,
    scale: float,
    *,
    grouped_qk: bool,
    storage_key_length: int,
    key_order: torch.Tensor | None,
    fuse_query_quantization: bool = False,
    fuse_key_value_quantization: bool = False,
    fuse_query_with_key_value: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    if (fuse_query_quantization or fuse_key_value_quantization) and grouped_qk:
        raise ValueError("fused Piper quantization currently requires SM89 per-row QK")
    query_int8 = (
        query
        if fuse_query_quantization
        else torch.empty(query.shape, device=query.device, dtype=torch.int8)
    )
    key_shape = (batch, heads, storage_key_length, head_dim)
    key_int8 = (
        torch.zeros(key_shape, device=key.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(key_shape, device=key.device, dtype=torch.int8)
    )
    if grouped_qk:
        query_groups = int(triton.cdiv(query_length, 32))
        key_groups = int(triton.cdiv(key_length, _BLOCK_N))
        query_scale = torch.empty(
            (batch, heads, query_groups),
            device=query.device,
            dtype=torch.float32,
        )
        key_scale = torch.empty(
            (batch, heads, key_groups),
            device=key.device,
            dtype=torch.float32,
        )
        qk_quantization.quantize_query_per_warp_kernel[(query_groups, heads, batch)](
            query,
            query_int8,
            query_scale,
            query_length,
            scale,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query_int8.stride(0),
            query_int8.stride(1),
            query_int8.stride(2),
            query_scale.stride(0),
            query_scale.stride(1),
            head_dim=head_dim,
            num_warps=4,
        )
        if key_order is None:
            qk_quantization.quantize_key_per_block_kernel[(key_groups, heads, batch)](
                key,
                key_mean,
                key_int8,
                key_scale,
                key_length,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key_int8.stride(0),
                key_int8.stride(1),
                key_int8.stride(2),
                key_scale.stride(0),
                key_scale.stride(1),
                heads=heads,
                head_dim=head_dim,
                block_n=_BLOCK_N,
                num_warps=4,
            )
        else:
            _quantize_ordered_key_per_block_kernel[(key_groups, heads, batch)](
                key,
                key_mean,
                key_order,
                key_int8,
                key_scale,
                key_length,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key_int8.stride(0),
                key_int8.stride(1),
                key_int8.stride(2),
                key_scale.stride(0),
                key_scale.stride(1),
                heads=heads,
                head_dim=head_dim,
                block_n=_BLOCK_N,
                num_warps=4,
            )
    else:
        if key_order is not None:
            raise ValueError("ordered K/V preparation requires grouped QK")
        query_scale = (
            key_mean
            if fuse_query_quantization
            else torch.empty(
                query.shape[:3],
                device=query.device,
                dtype=torch.float32,
            )
        )
        key_scale = torch.empty(key.shape[:3], device=key.device, dtype=torch.float32)
        if not fuse_query_quantization and not fuse_query_with_key_value:
            qk_quantization.quantize_query_per_thread_kernel[
                (triton.cdiv(query_length, 32) * 8, heads, batch)
            ](
                query,
                query_int8,
                query_scale,
                query_length,
                scale,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query_int8.stride(0),
                query_int8.stride(1),
                query_int8.stride(2),
                query_scale.stride(0),
                query_scale.stride(1),
                head_dim=head_dim,
                num_warps=4,
            )
        if not fuse_key_value_quantization:
            qk_quantization.quantize_key_per_thread_kernel[
                (triton.cdiv(key_length, _BLOCK_N) * 4, heads, batch)
            ](
                key,
                key_mean,
                key_int8,
                key_scale,
                key_length,
                key.stride(0),
                key.stride(1),
                key.stride(2),
                key_int8.stride(0),
                key_int8.stride(1),
                key_int8.stride(2),
                key_scale.stride(0),
                key_scale.stride(1),
                heads=heads,
                head_dim=head_dim,
                num_warps=4,
            )
    return query_int8, key_int8, query_scale, key_scale


def _make_descriptors(
    key: torch.Tensor,
    value: torch.Tensor,
    batch: int,
    heads: int,
    storage_key_length: int,
    head_dim: int,
    *,
    split_pv_head_dim: bool,
) -> tuple[TensorDescriptor, TensorDescriptor]:
    key_descriptor = TensorDescriptor(
        base=key,
        shape=[batch * heads, storage_key_length, head_dim],
        strides=[storage_key_length * head_dim, head_dim, 1],
        block_shape=[1, _BLOCK_N, head_dim],
    )
    value_descriptor = TensorDescriptor(
        base=value,
        shape=[batch * heads, head_dim, storage_key_length],
        strides=[head_dim * storage_key_length, storage_key_length, 1],
        block_shape=[
            1,
            head_dim // 2 if split_pv_head_dim else head_dim,
            _BLOCK_N,
        ],
    )
    return key_descriptor, value_descriptor


@dataclass(frozen=True, slots=True)
class _PreparedPiperAttention:
    query: torch.Tensor
    key: torch.Tensor | TensorDescriptor
    value: torch.Tensor | TensorDescriptor
    query_scale: torch.Tensor
    key_scale: torch.Tensor
    value_scale_multiplier: torch.Tensor
    value_log_scale: torch.Tensor
    value_correction: torch.Tensor
    value_mean: torch.Tensor
    output: torch.Tensor
    softmax_scale: float
    key_length: int
    storage_key_length: int
    block_m: int
    num_warps: int
    num_stages: int
    is_causal: bool
    grouped_qk: bool
    native_uint8: bool
    split_pv_head_dim: bool
    scaled_fp16_numerator: bool
    center_value: bool
    use_tensor_descriptors: bool
    reverse_causal_blocks: bool
    loop_num_stages: int | None
    disable_loop_licm: bool
    block_value_scale: bool
    round_probability_codes: bool
    use_packed_probability_conversion: bool
    use_sm89_d128_specialization: bool
    fuse_query_quantization: bool


def _should_sort_value_rows(
    *,
    center_value: bool,
    target: AcceleratorTarget,
    is_causal: bool,
    head_dim: int,
    key_length: int,
) -> bool:
    """Select exact V ordering only where its long-context cost is amortized."""
    return (
        center_value
        and target.is_cuda_capability(12)
        and not is_causal
        and head_dim == 128
        and key_length >= 16384
    )


def _select_sm89_schedule(
    *,
    default_block_m: int,
    batch: int,
    heads: int,
    query_length: int,
    head_dim: int,
    is_causal: bool,
    num_sms: int,
) -> tuple[int, int, int]:
    """Return the offline-tuned Ada launch schedule without hot-path autotuning."""
    parallelism = batch * heads

    def nearly_fills_device(candidate_block_m: int) -> bool:
        ctas = int(triton.cdiv(query_length, candidate_block_m)) * parallelism
        return 10 * ctas >= 9 * num_sms

    if is_causal and head_dim == 64:
        block_m = 64 if nearly_fills_device(64) else 32
        return block_m, 4, 4 if block_m == 64 else 3
    if is_causal:
        block_m = 128 if nearly_fills_device(128) else default_block_m
        if block_m == 128 and query_length >= 32768:
            return block_m, 4, 2
        if block_m == 128 and query_length >= 8192:
            return block_m, 4, 3
        return block_m, 8 if block_m == 128 else 4, 4 if block_m == 128 else 3

    block_m = default_block_m
    if head_dim == 64 and block_m == 32 and nearly_fills_device(64):
        block_m = 64
    num_stages = 2 if head_dim == 128 and block_m == 128 else 3
    return block_m, 4, num_stages


@dataclass(frozen=True, slots=True)
class _PiperLaunchSchedule:
    block_m: int
    num_warps: int
    num_stages: int
    use_tensor_descriptors: bool


def _default_piper_launch_schedule(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool,
) -> _PiperLaunchSchedule:
    """Resolve the frozen production schedule for benchmark metadata and launch."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    capability = torch.cuda.get_device_capability(query.device)
    nvidia_cuda = AcceleratorTarget.from_device(query.device).is_nvidia_cuda
    split_pv_head_dim = (
        nvidia_cuda
        and capability[0] == 12
        and not is_causal
        and head_dim == 128
        and query_length >= 1024
        and key_length >= 1024
    )
    scaled_fp16_numerator = split_pv_head_dim and key_length <= 131072
    block_m = (
        64
        if is_causal
        else 128
        if scaled_fp16_numerator and query_length >= 8192 and key_length >= 8192
        else 64
        if split_pv_head_dim
        else select_query_block(query, batch, heads, query_length)
    )
    num_warps = 4
    num_stages = 3
    if nvidia_cuda and capability == (8, 9):
        block_m, num_warps, num_stages = _select_sm89_schedule(
            default_block_m=select_query_block(query, batch, heads, query_length),
            batch=batch,
            heads=heads,
            query_length=query_length,
            head_dim=head_dim,
            is_causal=is_causal,
            num_sms=torch.cuda.get_device_properties(query.device).multi_processor_count,
        )
    padded_key_length = int(triton.cdiv(key_length, _BLOCK_N)) * _BLOCK_N
    use_tensor_descriptors = (
        nvidia_cuda
        and capability[0] == 12
        and block_m == 128
        and head_dim == 128
        and padded_key_length % 16 == 0
    )
    if use_tensor_descriptors:
        num_stages = 2
    return _PiperLaunchSchedule(
        block_m,
        num_warps,
        num_stages,
        use_tensor_descriptors,
    )


def _prepare_piper_attention(  # noqa: PLR0912, PLR0915 - execution-plan validation
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    center_value: bool,
    *,
    native_uint8: bool | None = None,
    sort_value_rows: bool | None = None,
    use_tensor_descriptors: bool | None = None,
    block_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_pv_head_dim: bool | None = None,
    scaled_fp16_numerator: bool | None = None,
    reverse_causal_blocks: bool | None = None,
    loop_num_stages: int | None = None,
    disable_loop_licm: bool | None = None,
    block_value_scale: bool | None = None,
    round_probability_codes: bool | None = None,
    use_packed_probability_conversion: bool | None = None,
    fuse_query_quantization: bool | None = None,
    fuse_query_with_key_value: bool | None = None,
) -> _PreparedPiperAttention:
    """Quantize Q/K/V and construct the selected launch specialization."""
    batch, heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    target = AcceleratorTarget.from_device(query.device)
    grouped_qk = target.is_cuda_capability(12)
    use_sm12x_schedule = target.is_cuda_capability(12)
    if native_uint8 is None:
        native_uint8 = target.supports_uint8_int8_mma
    if native_uint8:
        with torch.cuda.device(query.device):
            install_uint8_int8_dot_hook()
    default_split_pv_head_dim = (
        use_sm12x_schedule
        and not is_causal
        and head_dim == 128
        and query_length >= 1024
        and key_length >= 1024
    )
    sm89_long_d128 = (
        target.is_cuda_capability(8, 9)
        and head_dim == 128
        and query_length >= 8192
        and key_length >= 8192
    )
    if split_pv_head_dim is None:
        split_pv_head_dim = default_split_pv_head_dim or sm89_long_d128
    if split_pv_head_dim and head_dim != 128:
        raise ValueError("split-PV Piper Attention requires head_dim=128")
    if scaled_fp16_numerator is None:
        scaled_fp16_numerator = split_pv_head_dim and key_length <= 131072
    if reverse_causal_blocks is None:
        reverse_causal_blocks = sm89_long_d128 and is_causal
    if disable_loop_licm is None:
        disable_loop_licm = not (sm89_long_d128 and not is_causal)
    if block_value_scale is None:
        block_value_scale = sm89_long_d128
    if round_probability_codes is None:
        round_probability_codes = not (
            sm89_long_d128
            and not is_causal
            and block_value_scale
            and key_length >= 32768
        )
    if sort_value_rows is None:
        sort_value_rows = _should_sort_value_rows(
            center_value=center_value,
            target=target,
            is_causal=is_causal,
            head_dim=head_dim,
            key_length=key_length,
        )
    if sort_value_rows and (not center_value or is_causal or not grouped_qk):
        raise ValueError(
            "value-row ordering requires centered non-causal grouped-QK attention"
        )

    default_schedule = _default_piper_launch_schedule(query, key, is_causal)
    block_m = default_schedule.block_m if block_m is None else block_m
    if block_m not in (32, 64, 128):
        raise ValueError(f"Piper Attention block_m must be 32, 64, or 128, got {block_m}")
    padded_key_length = int(triton.cdiv(key_length, _BLOCK_N)) * _BLOCK_N
    if use_tensor_descriptors is None:
        use_tensor_descriptors = default_schedule.use_tensor_descriptors and (
            block_m == default_schedule.block_m
        )
    if use_tensor_descriptors and block_m != 128:
        raise ValueError("tensor-descriptor Piper Attention requires block_m=128")
    num_warps = default_schedule.num_warps if num_warps is None else num_warps
    if num_warps not in (4, 8):
        raise ValueError(f"Piper Attention num_warps must be 4 or 8, got {num_warps}")
    default_num_stages = 2 if use_tensor_descriptors else default_schedule.num_stages
    num_stages = default_num_stages if num_stages is None else num_stages
    if num_stages not in (2, 3, 4):
        raise ValueError(f"Piper Attention num_stages must be 2, 3, or 4, got {num_stages}")
    if loop_num_stages is None:
        loop_num_stages = (
            2
            if sm89_long_d128 and not is_causal and key_length >= 131072
            else num_stages
        )
    if loop_num_stages not in (1, 2, 3, 4):
        raise ValueError("Piper loop_num_stages must be 1, 2, 3, or 4")
    storage_key_length = padded_key_length if use_tensor_descriptors else key_length
    use_sm89_d128_specialization = (
        sm89_long_d128
        and query_length == key_length
        and query_length % 128 == 0
        and key_length % _BLOCK_N == 0
        and block_m == 128
        and native_uint8
        and split_pv_head_dim
        and scaled_fp16_numerator
        and not use_tensor_descriptors
    )
    if fuse_query_quantization is None:
        fuse_query_quantization = (
            use_sm89_d128_specialization and key_length < 32768
        )
    if fuse_query_quantization and not use_sm89_d128_specialization:
        raise ValueError("fused Piper query quantization requires the SM89 D128 path")
    if fuse_query_with_key_value is None:
        fuse_query_with_key_value = (
            use_sm89_d128_specialization and not fuse_query_quantization
        )
    if use_packed_probability_conversion is None:
        use_packed_probability_conversion = (
            use_sm89_d128_specialization
            and block_value_scale
        )
    if fuse_query_with_key_value and not use_sm89_d128_specialization:
        raise ValueError("fused Piper Q/K/V preparation requires the SM89 D128 path")
    if fuse_query_quantization and fuse_query_with_key_value:
        raise ValueError("Piper query quantization can only be fused into one kernel")
    key_mean, value_mean = _compute_kv_means(
        key,
        value,
        center_value=center_value,
        value_mean_dtype=(query.dtype if use_sm89_d128_specialization else torch.float32),
    )
    key_order = (
        _build_centered_value_order(value, value_mean)
        if sort_value_rows
        else None
    )
    query_int8, key_int8, query_scale, key_scale = _prepare_qk(
        query,
        key,
        key_mean,
        scale,
        grouped_qk=grouped_qk,
        storage_key_length=storage_key_length,
        key_order=key_order,
        fuse_query_quantization=fuse_query_quantization,
        fuse_key_value_quantization=use_sm89_d128_specialization,
        fuse_query_with_key_value=fuse_query_with_key_value,
    )

    value_shape = (batch, heads, head_dim, storage_key_length)
    value_int8 = (
        torch.zeros(value_shape, device=value.device, dtype=torch.int8)
        if storage_key_length != key_length
        else torch.empty(value_shape, device=value.device, dtype=torch.int8)
    )
    value_scale_multiplier = (
        key_scale
        if block_value_scale
        else torch.empty(
            (batch, heads, key_length),
            device=value.device,
            dtype=torch.float32,
        )
    )
    value_log_scale = torch.empty(
        (
            (batch, heads, int(triton.cdiv(key_length, _BLOCK_N)))
            if block_value_scale
            else (batch, heads, key_length)
        ),
        device=value.device,
        dtype=torch.float32 if block_value_scale else torch.float16,
    )
    value_correction = (
        torch.empty(
            (batch, heads, int(triton.cdiv(key_length, _BLOCK_N)), head_dim),
            device=value.device,
            dtype=torch.int16,
        )
        if not native_uint8
        else key_scale
    )
    order_argument = (
        key_order if key_order is not None else key_int8
    )
    _quantize_value_per_key_kernel[
        (triton.cdiv(key_length, _BLOCK_N), heads, batch)
    ](
        value,
        key,
        query,
        value_mean,
        key_mean,
        order_argument,
        value_scale_multiplier,
        value_log_scale,
        value_correction,
        value_int8,
        key_int8,
        key_scale,
        query_int8,
        query_scale,
        key_length,
        storage_key_length,
        scale,
        value.stride(0),
        value.stride(1),
        value.stride(2),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        value_int8.stride(0),
        value_int8.stride(1),
        value_int8.stride(2),
        value_int8.stride(3),
        center_value=center_value,
        ordered=key_order is not None,
        store_correction=not native_uint8,
        block_value_scale=block_value_scale,
        fuse_key_quantization=use_sm89_d128_specialization,
        fuse_query_quantization=fuse_query_with_key_value,
        heads=heads,
        head_dim=head_dim,
        block_n=_BLOCK_N,
        num_warps=4,
    )

    key_argument: torch.Tensor | TensorDescriptor = key_int8
    value_argument: torch.Tensor | TensorDescriptor = value_int8
    if use_tensor_descriptors:
        key_argument, value_argument = _make_descriptors(
            key_int8,
            value_int8,
            batch,
            heads,
            storage_key_length,
            head_dim,
            split_pv_head_dim=split_pv_head_dim,
        )
    output = torch.empty(query.shape, device=query.device, dtype=query.dtype)
    return _PreparedPiperAttention(
        query=query_int8,
        key=key_argument,
        value=value_argument,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale_multiplier=value_scale_multiplier,
        value_log_scale=value_log_scale,
        value_correction=value_correction,
        value_mean=value_mean,
        output=output,
        softmax_scale=scale,
        key_length=key_length,
        storage_key_length=storage_key_length,
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
        is_causal=is_causal,
        grouped_qk=grouped_qk,
        native_uint8=native_uint8,
        split_pv_head_dim=split_pv_head_dim,
        scaled_fp16_numerator=scaled_fp16_numerator,
        center_value=center_value,
        use_tensor_descriptors=use_tensor_descriptors,
        reverse_causal_blocks=reverse_causal_blocks,
        loop_num_stages=loop_num_stages,
        disable_loop_licm=disable_loop_licm,
        block_value_scale=block_value_scale,
        round_probability_codes=round_probability_codes,
        use_packed_probability_conversion=use_packed_probability_conversion,
        use_sm89_d128_specialization=use_sm89_d128_specialization,
        fuse_query_quantization=fuse_query_quantization,
    )


def _launch_piper_attention(prepared: _PreparedPiperAttention) -> torch.Tensor:
    """Launch only the fused attention recurrence on prepared integer inputs."""
    batch, heads, query_length, head_dim = prepared.output.shape
    attention_kernel = cast(Any, _piper_attention_kernel)
    launch_options = {
        "num_warps": prepared.num_warps,
        "num_stages": prepared.num_stages,
    }
    if prepared.use_sm89_d128_specialization:
        sm89_kernel = cast(Any, _piper_attention_sm89_d128_kernel)
        sm89_kernel[(query_length // prepared.block_m, heads, batch)](
            prepared.query,
            cast(torch.Tensor, prepared.key),
            cast(torch.Tensor, prepared.value),
            prepared.query_scale,
            prepared.key_scale,
            prepared.value_scale_multiplier,
            prepared.value_log_scale,
            prepared.value_mean,
            prepared.output,
            query_length,
            prepared.key_length,
            prepared.softmax_scale,
            is_causal=prepared.is_causal,
            reverse_causal_blocks=prepared.reverse_causal_blocks,
            block_value_scale=prepared.block_value_scale,
            round_probability_codes=prepared.round_probability_codes,
            use_packed_probability_conversion=prepared.use_packed_probability_conversion,
            center_value=prepared.center_value,
            fuse_query_quantization=prepared.fuse_query_quantization,
            loop_num_stages=prepared.loop_num_stages,
            disable_loop_licm=prepared.disable_loop_licm,
            heads=heads,
            block_m=prepared.block_m,
            block_n=_BLOCK_N,
            **launch_options,
        )
        return prepared.output

    def launch(query_blocks: int, query_block_offset: int, unmasked_queries: bool) -> None:
        attention_kernel[(query_blocks, heads, batch)](
            prepared.query,
            prepared.key,
            prepared.value,
            prepared.query_scale,
            prepared.key_scale,
            prepared.value_scale_multiplier,
            prepared.value_log_scale,
            prepared.value_correction,
            prepared.value_mean,
            prepared.output,
            query_length,
            prepared.key_length,
            prepared.storage_key_length,
            query_block_offset=query_block_offset,
            is_causal=prepared.is_causal,
            grouped_qk=prepared.grouped_qk,
            native_uint8=prepared.native_uint8,
            split_pv_head_dim=prepared.split_pv_head_dim,
            scaled_fp16_numerator=prepared.scaled_fp16_numerator,
            center_value=prepared.center_value,
            unmasked_query_tiles=unmasked_queries,
            unmasked_self_attention=(
                unmasked_queries
                and not prepared.is_causal
                and query_length == prepared.key_length
                and prepared.key_length % _BLOCK_N == 0
            ),
            heads=heads,
            head_dim=head_dim,
            block_m=prepared.block_m,
            block_n=_BLOCK_N,
            use_tensor_descriptors=prepared.use_tensor_descriptors,
            reverse_causal_blocks=prepared.reverse_causal_blocks,
            loop_num_stages=prepared.loop_num_stages,
            disable_loop_licm=prepared.disable_loop_licm,
            block_value_scale=prepared.block_value_scale,
            round_probability_codes=prepared.round_probability_codes,
            **launch_options,
        )

    full_query_blocks = query_length // prepared.block_m
    if full_query_blocks:
        launch(full_query_blocks, 0, True)
    if query_length % prepared.block_m:
        launch(1, full_query_blocks, False)
    return prepared.output


def _run_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    center_value: bool,
    *,
    native_uint8: bool | None = None,
    sort_value_rows: bool | None = None,
    use_tensor_descriptors: bool | None = None,
    block_m: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused recurrence."""
    prepared = _prepare_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        center_value,
        native_uint8=native_uint8,
        sort_value_rows=sort_value_rows,
        use_tensor_descriptors=use_tensor_descriptors,
        block_m=block_m,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return _launch_piper_attention(prepared)


@torch.library.custom_op("piper_kernels::piper_attention", mutates_args=())
def triton_piper_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    is_causal: bool,
    center_value: bool,
) -> torch.Tensor:
    """Run Piper Attention preprocessing and its fused integer-PV kernel."""
    return _run_piper_attention(
        query,
        key,
        value,
        scale,
        is_causal,
        center_value,
    )


@triton_piper_attention.register_fake
def _triton_piper_attention_fake(
    query: torch.Tensor,
    _key: torch.Tensor,
    _value: torch.Tensor,
    _scale: float,
    _is_causal: bool,
    _center_value: bool,
) -> torch.Tensor:
    return torch.empty_like(query, memory_format=torch.contiguous_format)
