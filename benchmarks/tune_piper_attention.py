"""Tune a small Piper Attention schedule search offline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import torch
from lib.attention import (
    AttentionConfig,
    AttentionInputs,
    AttentionShape,
    attention_dtype,
    make_attention_inputs,
    run_sdpa,
)
from lib.attention_tuning import (
    add_attention_tuning_arguments,
    validate_attention_tuning_arguments,
)
from lib.environment import capture_environment
from lib.providers import BenchmarkProvider
from lib.quality import measure_quality
from lib.reporting import output_target
from lib.tuning import (
    TuningCandidate,
    UnsupportedTuningCandidateError,
    boolean_tuning_axis,
    meets_minimum_sqnr,
    report_tuning_run,
    tune_candidates,
    tuning_axis,
    validate_tuning_candidate_count,
)

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _policy as piper_attention_policy
from piper_kernels.attention.piper_attention import triton as piper_attention_backend


def _validate_args(arguments: argparse.Namespace) -> None:
    """Validate shared attention controls and Piper-only causal traversal."""
    validate_attention_tuning_arguments(arguments)
    if not arguments.causal and arguments.optimize_causal_traversal is True:
        raise SystemExit("optimized causal traversal requires causal attention")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_attention_tuning_arguments(parser, include_reverse_causal_blocks=False)
    parser.add_argument(
        "--derive-value-log-bound",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="derive the V log-scale bound from its FP32 multiplier; omitted retains production",
    )
    parser.add_argument(
        "--optimize-causal-traversal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use reverse CTA order with an unmasked prefix and masked boundary",
    )
    parser.add_argument(
        "--use-sm89-d128-specialization",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-shared-value-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-fused-kv-preprocessing",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-fp16-value-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--derive-value-scale-multiplier",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-hybrid-fp32-fp16-numerator",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--use-strided-kv-mean-sample",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--scaled-fp16-numerator",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--round-probability-codes",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _candidate_plans(  # noqa: PLR0912 - normalizes dependent plan axes
    args: argparse.Namespace,
    production_plan: piper_attention_policy.PiperAttentionExecutionPlan,
) -> tuple[piper_attention_policy.PiperAttentionExecutionPlan, ...]:
    """Build an explicit bounded search around production policy."""
    axes = (
        tuning_axis(args.block_m, production_plan.block_m),
        tuning_axis(args.num_warps, production_plan.num_warps),
        tuning_axis(args.num_stages, production_plan.num_stages),
        boolean_tuning_axis(args.use_tensor_descriptors, production_plan.use_tensor_descriptors),
        tuning_axis(args.loop_num_stages, production_plan.loop_num_stages),
        boolean_tuning_axis(args.loop_licm, production_plan.loop_licm),
        boolean_tuning_axis(
            args.use_packed_probability_conversion,
            production_plan.use_packed_probability_conversion,
        ),
        boolean_tuning_axis(args.derive_value_log_bound, production_plan.derive_value_log_bound),
        boolean_tuning_axis(
            args.optimize_causal_traversal,
            production_plan.optimize_causal_traversal,
        ),
        boolean_tuning_axis(
            args.use_sm89_d128_specialization,
            production_plan.use_sm89_d128_specialization,
        ),
        boolean_tuning_axis(args.use_shared_value_scale, production_plan.use_shared_value_scale),
        boolean_tuning_axis(
            args.use_fused_kv_preprocessing,
            production_plan.use_fused_kv_preprocessing,
        ),
        boolean_tuning_axis(args.use_fp16_value_scale, production_plan.use_fp16_value_scale),
        boolean_tuning_axis(
            args.derive_value_scale_multiplier,
            production_plan.derive_value_scale_multiplier,
        ),
        boolean_tuning_axis(
            args.use_hybrid_fp32_fp16_numerator,
            production_plan.use_hybrid_fp32_fp16_numerator,
        ),
        boolean_tuning_axis(
            args.use_strided_kv_mean_sample,
            production_plan.use_strided_kv_mean_sample,
        ),
        boolean_tuning_axis(args.scaled_fp16_numerator, production_plan.scaled_fp16_numerator),
        boolean_tuning_axis(args.round_probability_codes, production_plan.round_probability_codes),
    )
    plans: list[piper_attention_policy.PiperAttentionExecutionPlan] = []
    for values in product(*axes):
        (
            block_m,
            num_warps,
            num_stages,
            use_tensor_descriptors,
            loop_num_stages,
            loop_licm,
            use_packed_probability_conversion,
            derive_value_log_bound,
            optimize_causal_traversal,
            use_sm89_d128_specialization,
            use_shared_value_scale,
            use_fused_kv_preprocessing,
            use_fp16_value_scale,
            derive_value_scale_multiplier,
            use_hybrid_fp32_fp16_numerator,
            use_strided_kv_mean_sample,
            scaled_fp16_numerator,
            round_probability_codes,
        ) = values
        if use_shared_value_scale and args.use_fp16_value_scale is None:
            use_fp16_value_scale = False
        if (
            use_shared_value_scale or not use_fused_kv_preprocessing or not use_fp16_value_scale
        ) and args.derive_value_scale_multiplier is None:
            derive_value_scale_multiplier = False
        if use_hybrid_fp32_fp16_numerator and args.scaled_fp16_numerator is None:
            scaled_fp16_numerator = False
        if scaled_fp16_numerator and args.use_hybrid_fp32_fp16_numerator is None:
            use_hybrid_fp32_fp16_numerator = False
        if (
            use_shared_value_scale or use_fp16_value_scale or derive_value_scale_multiplier
        ) and args.use_hybrid_fp32_fp16_numerator is None:
            use_hybrid_fp32_fp16_numerator = False

        if not use_sm89_d128_specialization:
            if args.use_shared_value_scale is None:
                use_shared_value_scale = False
            if args.use_fused_kv_preprocessing is None:
                use_fused_kv_preprocessing = False
            if args.use_fp16_value_scale is None:
                use_fp16_value_scale = False
            if args.derive_value_scale_multiplier is None:
                derive_value_scale_multiplier = False
            if args.use_hybrid_fp32_fp16_numerator is None:
                use_hybrid_fp32_fp16_numerator = False
            if args.use_strided_kv_mean_sample is None:
                use_strided_kv_mean_sample = False
            if args.scaled_fp16_numerator is None:
                scaled_fp16_numerator = False
            if args.round_probability_codes is None:
                round_probability_codes = True
        elif args.derive_value_log_bound is None:
            derive_value_log_bound = False

        try:
            plan = replace(
                production_plan,
                block_m=block_m,
                num_warps=num_warps,
                num_stages=num_stages,
                use_tensor_descriptors=False
                if use_sm89_d128_specialization and args.use_tensor_descriptors is None
                else use_tensor_descriptors,
                loop_num_stages=loop_num_stages,
                loop_licm=loop_licm,
                use_packed_probability_conversion=use_packed_probability_conversion,
                derive_value_log_bound=derive_value_log_bound,
                optimize_causal_traversal=optimize_causal_traversal,
                split_pv_head_dim=(
                    True if use_sm89_d128_specialization else production_plan.split_pv_head_dim
                ),
                scaled_fp16_numerator=scaled_fp16_numerator,
                use_sm89_d128_specialization=use_sm89_d128_specialization,
                use_shared_value_scale=use_shared_value_scale,
                use_fused_kv_preprocessing=use_fused_kv_preprocessing,
                use_fp16_value_scale=use_fp16_value_scale,
                derive_value_scale_multiplier=derive_value_scale_multiplier,
                use_hybrid_fp32_fp16_numerator=use_hybrid_fp32_fp16_numerator,
                use_strided_kv_mean_sample=use_strided_kv_mean_sample,
                round_probability_codes=round_probability_codes,
            )
        except ValueError:
            continue
        if plan not in plans:
            plans.append(plan)
    validate_tuning_candidate_count((plans,), args.max_candidates)
    return tuple(plans)


def _plan_name(plan: piper_attention_policy.PiperAttentionExecutionPlan) -> str:
    load_path = "descriptor" if plan.use_tensor_descriptors else "pointer"
    loop_stages = plan.loop_num_stages if plan.loop_num_stages is not None else "default"
    licm = "licm" if plan.loop_licm else "no-licm"
    probability_conversion = "packed-p" if plan.use_packed_probability_conversion else "stock-p"
    value_metadata = "derived-vlog" if plan.derive_value_log_bound else "stored-vlog"
    causal_traversal = "optimized-causal" if plan.optimize_causal_traversal else "monolithic"
    kernel = "sm89-d128" if plan.use_sm89_d128_specialization else "generic"
    value_scale = "shared-v64" if plan.use_shared_value_scale else "per-key-v"
    accumulator = "split-fp16" if plan.scaled_fp16_numerator else "fp32"
    preprocessing = "fused-qkv" if plan.use_fused_kv_preprocessing else "stock-prep"
    value_scale_storage = "fp16-vscale" if plan.use_fp16_value_scale else "fp32-vscale"
    value_scale_load = "derived-vscale" if plan.derive_value_scale_multiplier else "loaded-vscale"
    numerator = "hybrid-acc" if plan.use_hybrid_fp32_fp16_numerator else accumulator
    mean_preprocessing = "sampled-mean" if plan.use_strided_kv_mean_sample else "exact-mean"
    probability_rounding = "round-p" if plan.round_probability_codes else "truncate-p"
    return (
        f"{kernel}-{load_path}-m{plan.block_m}-w{plan.num_warps}-s{plan.num_stages}-"
        f"loop{loop_stages}-{licm}-{probability_conversion}-{value_metadata}-"
        f"{causal_traversal}-"
        f"{value_scale}-{value_scale_storage}-{value_scale_load}-{numerator}-{preprocessing}-"
        f"{mean_preprocessing}-"
        f"{probability_rounding}"
    )


def _make_candidate(
    plan: piper_attention_policy.PiperAttentionExecutionPlan,
    inputs: AttentionInputs,
    *,
    config: AttentionConfig,
    target: AcceleratorTarget,
) -> TuningCandidate[object, torch.Tensor]:
    name = _plan_name(plan)
    query, key, value = inputs
    scale = config.scale if config.scale is not None else query.shape[-1] ** -0.5

    def make_provider() -> BenchmarkProvider[object, torch.Tensor]:
        if plan.use_tensor_descriptors and not target.is_cuda_capability(12):
            raise UnsupportedTuningCandidateError(
                "the tensor-descriptor candidate currently targets SM12x"
            )

        def prepare() -> object:
            return piper_attention_backend._prepare_piper_attention(
                query,
                key,
                value,
                scale,
                config.is_causal,
                execution_plan=plan,
            )

        def run(prepared: object) -> torch.Tensor:
            return piper_attention_backend._launch_piper_attention(
                cast(piper_attention_backend._PreparedPiperAttention, prepared)
            )

        return BenchmarkProvider(
            name=f"piper_attention_{name.replace('-', '_')}",
            prepare=prepare,
            run=run,
            synchronize=torch.cuda.synchronize,
        )

    return TuningCandidate(
        name=name,
        configuration={
            **config.as_dict(),
            "algorithm": "piper_attention",
            **plan.as_dict(),
        },
        make_provider=make_provider,
    )


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit("Piper Attention tuning requires an available NVIDIA GPU")
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    if not target.supports_uint8_int8_mma:
        raise SystemExit("Piper Attention tuning requires NVIDIA SM8x or SM12x")

    key_value_length = args.kv_sequence or args.sequence
    shape = AttentionShape(
        args.batch_size,
        args.heads,
        args.sequence,
        key_value_length,
        args.head_dim,
    )
    scale = args.head_dim**-0.5
    config = AttentionConfig(
        dtype=attention_dtype(args.dtype),
        is_causal=args.causal,
        scale=scale,
        seed=args.seed,
    )
    inputs = make_attention_inputs(shape, config=config, device=device)
    query, key, _ = inputs
    production_plan = piper_attention_backend._default_piper_attention_execution_plan(
        query,
        args.causal,
        key_length=key.shape[2],
        target=target,
    )
    candidates = tuple(
        _make_candidate(
            plan,
            inputs,
            config=config,
            target=target,
        )
        for plan in _candidate_plans(args, production_plan)
    )
    expected = run_sdpa(inputs, config)
    run = tune_candidates(
        candidates,
        tuning="piper_attention_execution_plan",
        shape=shape.as_dict(),
        environment=capture_environment(Path(__file__).resolve().parents[1]),
        phase=args.phase,
        warmup_ms=args.warmup_ms,
        measurement_time_ms=args.measurement_time_ms,
        measure_candidate_quality=lambda output: measure_quality(output, expected),
        quality_gate=lambda quality: meets_minimum_sqnr(quality, args.minimum_sqnr_db),
    )
    report_tuning_run(run, output_target(args))


def main() -> None:
    """Run the Piper Attention schedule search."""
    _main()


if __name__ == "__main__":
    main()
