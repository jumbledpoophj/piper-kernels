"""Backend-independent execution planning for Piper Attention."""

from dataclasses import asdict, dataclass

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.scheduling import (
    BLOCK_M_VALUES,
    LOOP_NUM_STAGES_VALUES,
    NUM_STAGES_VALUES,
    NUM_WARPS_VALUES,
)


@dataclass(frozen=True, slots=True)
class PiperAttentionExecutionPlan:
    """Host-side specialization and launch choices for one Piper invocation."""

    block_m: int
    grouped_qk: bool
    split_pv_head_dim: bool
    use_tensor_descriptors: bool
    derive_value_log_bound: bool = False
    optimize_causal_traversal: bool = False
    num_warps: int = 4
    num_stages: int = 3
    loop_num_stages: int | None = None
    loop_licm: bool = False
    use_packed_probability_conversion: bool = False
    scaled_fp16_numerator: bool = False
    use_sm89_d128_specialization: bool = False
    use_shared_value_scale: bool = False
    use_fused_kv_preprocessing: bool = False
    use_fp16_value_scale: bool = False
    derive_value_scale_multiplier: bool = False
    use_hybrid_fp32_fp16_numerator: bool = False
    use_strided_kv_mean_sample: bool = False
    round_probability_codes: bool = True

    def __post_init__(self) -> None:  # noqa: PLR0912 - plan invariants stay explicit
        if self.block_m not in BLOCK_M_VALUES:
            raise ValueError("Piper Attention block_m must be 64 or 128")
        if self.num_warps not in NUM_WARPS_VALUES:
            raise ValueError("Piper Attention num_warps must be 2, 4, or 8")
        if self.num_stages not in NUM_STAGES_VALUES:
            raise ValueError("Piper Attention num_stages must be 1, 2, 3, or 4")
        if self.loop_num_stages not in LOOP_NUM_STAGES_VALUES:
            raise ValueError("Piper Attention loop_num_stages must be None, 1, 2, 3, or 4")
        if self.scaled_fp16_numerator and not self.split_pv_head_dim:
            raise ValueError("scaled FP16 numerator recurrence requires split PV")
        if self.use_sm89_d128_specialization and not self.split_pv_head_dim:
            raise ValueError("SM89 D128 specialization requires split PV")
        if self.use_sm89_d128_specialization and self.use_tensor_descriptors:
            raise ValueError("SM89 D128 specialization requires pointer loads")
        if self.use_shared_value_scale and not self.use_sm89_d128_specialization:
            raise ValueError("shared V scaling requires the SM89 D128 specialization")
        if self.use_fused_kv_preprocessing and not self.use_sm89_d128_specialization:
            raise ValueError("fused Q/K/V preprocessing requires the SM89 D128 specialization")
        if self.use_fp16_value_scale and not self.use_sm89_d128_specialization:
            raise ValueError("FP16 V-scale storage requires the SM89 D128 specialization")
        if self.use_fp16_value_scale and self.use_shared_value_scale:
            raise ValueError("FP16 per-key V-scale storage is incompatible with shared V scaling")
        if self.derive_value_scale_multiplier and not self.use_sm89_d128_specialization:
            raise ValueError("derived V-scale reconstruction requires the SM89 D128 specialization")
        if self.derive_value_scale_multiplier and not self.use_fused_kv_preprocessing:
            raise ValueError("derived V-scale reconstruction requires fused Q/K/V preprocessing")
        if self.derive_value_scale_multiplier and not self.use_fp16_value_scale:
            raise ValueError("derived V-scale reconstruction requires FP16 V-scale storage")
        if self.derive_value_scale_multiplier and self.use_shared_value_scale:
            raise ValueError("derived V-scale reconstruction requires per-key V scaling")
        if self.use_hybrid_fp32_fp16_numerator and not self.use_sm89_d128_specialization:
            raise ValueError("hybrid numerator accumulation requires the SM89 D128 specialization")
        if self.use_hybrid_fp32_fp16_numerator and self.scaled_fp16_numerator:
            raise ValueError("hybrid numerator accumulation requires the FP32 recurrence path")
        if self.use_hybrid_fp32_fp16_numerator and self.use_shared_value_scale:
            raise ValueError("hybrid numerator accumulation requires per-key V scaling")
        if self.use_hybrid_fp32_fp16_numerator and self.use_fp16_value_scale:
            raise ValueError("hybrid numerator accumulation requires FP32 V-scale storage")
        if self.use_hybrid_fp32_fp16_numerator and self.derive_value_scale_multiplier:
            raise ValueError("hybrid numerator accumulation requires loaded V-scale multipliers")
        if self.use_strided_kv_mean_sample and not self.use_sm89_d128_specialization:
            raise ValueError("strided K/V mean sampling requires the SM89 D128 specialization")
        if not self.round_probability_codes and not self.use_sm89_d128_specialization:
            raise ValueError("probability truncation requires the SM89 D128 specialization")

    def as_dict(self) -> dict[str, object]:
        """Return execution choices as serializable benchmark metadata."""
        return asdict(self)


def _generic_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Build capability-based defaults before exact-target tuning is applied."""
    grouped_qk = target.is_cuda_capability(12)
    split_pv_head_dim = target.is_cuda_capability(12) and not is_causal and head_dim == 128
    block_m = 64 if is_causal or split_pv_head_dim else 128
    use_tensor_descriptors = target.is_cuda_capability(12) and block_m == 128 and head_dim == 128
    return PiperAttentionExecutionPlan(
        block_m=block_m,
        grouped_qk=grouped_qk,
        split_pv_head_dim=split_pv_head_dim,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors else 3,
    )


def _sm89_execution_plan(
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int | None,
    key_length: int | None,
) -> PiperAttentionExecutionPlan:
    """Build the exact-SM89 plan, including the measured aligned D128 path."""
    noncausal_d128 = not is_causal and head_dim == 128
    aligned_d128 = (
        head_dim == 128
        and query_length is not None
        and key_length is not None
        and query_length == key_length
        and query_length % 128 == 0
        and key_length % 64 == 0
    )
    if not aligned_d128:
        return PiperAttentionExecutionPlan(
            block_m=64 if is_causal else 128,
            grouped_qk=False,
            split_pv_head_dim=noncausal_d128,
            use_tensor_descriptors=False,
            num_stages=1 if noncausal_d128 else 3,
            loop_num_stages=3 if noncausal_d128 else None,
            loop_licm=noncausal_d128,
            use_packed_probability_conversion=noncausal_d128,
        )

    return PiperAttentionExecutionPlan(
        block_m=128,
        grouped_qk=False,
        split_pv_head_dim=True,
        use_tensor_descriptors=False,
        optimize_causal_traversal=is_causal,
        num_warps=4,
        num_stages=1,
        loop_num_stages=3,
        loop_licm=not is_causal,
        use_packed_probability_conversion=True,
        scaled_fp16_numerator=False,
        use_sm89_d128_specialization=True,
        use_fused_kv_preprocessing=True,
        use_fp16_value_scale=False,
        derive_value_scale_multiplier=False,
        use_hybrid_fp32_fp16_numerator=True,
    )


def _sm120_execution_plan(
    *,
    head_dim: int,
    is_causal: bool,
) -> PiperAttentionExecutionPlan:
    """Build the loop, probability, and value-metadata plan measured on exact SM120."""
    split_pv_head_dim = head_dim == 128
    use_tensor_descriptors = head_dim == 128 and not is_causal
    return PiperAttentionExecutionPlan(
        block_m=64 if is_causal else 128,
        grouped_qk=True,
        split_pv_head_dim=split_pv_head_dim,
        use_tensor_descriptors=use_tensor_descriptors,
        num_stages=2 if use_tensor_descriptors else 3,
        derive_value_log_bound=not is_causal,
        optimize_causal_traversal=is_causal,
        use_packed_probability_conversion=not (is_causal and head_dim == 128),
    )


def select_execution_plan(
    target: AcceleratorTarget,
    *,
    head_dim: int,
    is_causal: bool,
    query_length: int | None = None,
    key_length: int | None = None,
) -> PiperAttentionExecutionPlan:
    """Combine portable capability defaults with exact-target measured policy."""
    if target.is_cuda_capability(8, 9):
        return _sm89_execution_plan(
            head_dim=head_dim,
            is_causal=is_causal,
            query_length=query_length,
            key_length=key_length,
        )
    if target.is_cuda_capability(12, 0):
        return _sm120_execution_plan(
            head_dim=head_dim,
            is_causal=is_causal,
        )
    return _generic_execution_plan(
        target,
        head_dim=head_dim,
        is_causal=is_causal,
    )
