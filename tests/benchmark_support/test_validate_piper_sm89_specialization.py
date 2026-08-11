from validate_piper_sm89_specialization import (
    _ablation_plans,
    _generic_plan,
    _parse_args,
    _validate_args,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention._policy import select_execution_plan


def _production_plan():
    return select_execution_plan(
        AcceleratorTarget(backend="cuda", architecture="sm89"),
        candidate_block_m=128,
        query_length=8192,
        key_length=8192,
        head_dim=128,
        is_causal=False,
    )


def test_validator_defaults_cover_all_requested_shapes_and_multiple_seeds() -> None:
    arguments = _parse_args([])

    assert tuple(arguments.sequences) == (8192, 32768, 131072)
    assert tuple(arguments.seeds) == (0, 1, 2)
    assert arguments.heads == 8
    _validate_args(arguments)


def test_generic_control_disables_every_specialized_axis() -> None:
    plan = _generic_plan(_production_plan(), packed_probability=False)

    assert not plan.use_sm89_d128_specialization
    assert not plan.split_pv_head_dim
    assert not plan.scaled_fp16_numerator
    assert not plan.use_shared_value_scale
    assert not plan.use_fused_kv_preprocessing
    assert not plan.use_fp16_value_scale
    assert not plan.derive_value_scale_multiplier
    assert not plan.use_hybrid_fp32_fp16_numerator
    assert not plan.use_packed_probability_conversion
    assert plan.round_probability_codes


def test_ablation_matrix_separates_requested_variables() -> None:
    plans = dict(_ablation_plans(_production_plan()))

    assert set(plans) == {
        "generic-stock",
        "generic-packed",
        "dedicated-stock-per-key-fp32-acc-fp32-scale-unfused-round",
        "dedicated-packed-per-key-fp32-acc-fp32-scale-unfused-round",
        "dedicated-packed-per-key-fp32-acc-fp16-scale-unfused-round",
        "dedicated-packed-per-key-split-fp16-fp16-scale-unfused-round",
        "dedicated-packed-per-key-production-acc-fp16-scale-fused-loaded-round",
        "production",
        "dedicated-packed-shared-v64-production-acc-fp32-scale-fused-round",
        "dedicated-packed-per-key-production-acc-fp16-scale-fused-truncate",
    }
    assert not plans["generic-stock"].use_packed_probability_conversion
    assert plans["generic-packed"].use_packed_probability_conversion
    assert not plans[
        "dedicated-stock-per-key-fp32-acc-fp32-scale-unfused-round"
    ].scaled_fp16_numerator
    assert not plans[
        "dedicated-packed-per-key-fp32-acc-fp32-scale-unfused-round"
    ].use_fp16_value_scale
    assert plans["dedicated-packed-per-key-fp32-acc-fp16-scale-unfused-round"].use_fp16_value_scale
    assert plans[
        "dedicated-packed-per-key-split-fp16-fp16-scale-unfused-round"
    ].scaled_fp16_numerator
    assert plans["production"].use_fused_kv_preprocessing
    assert plans["production"].derive_value_scale_multiplier
    assert not plans[
        "dedicated-packed-per-key-production-acc-fp16-scale-fused-loaded-round"
    ].derive_value_scale_multiplier
    assert plans[
        "dedicated-packed-shared-v64-production-acc-fp32-scale-fused-round"
    ].use_shared_value_scale
    assert not plans[
        "dedicated-packed-per-key-production-acc-fp16-scale-fused-truncate"
    ].round_probability_codes
