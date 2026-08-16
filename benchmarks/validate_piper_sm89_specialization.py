"""Validate the production SM89/D128 Piper specialization and its ablations."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from lib.attention import AttentionConfig, AttentionShape, make_attention_inputs, run_sdpa
from lib.environment import capture_environment
from lib.quality import measure_quality, measure_saturation
from lib.timing import synchronized_wall_benchmark, triton_benchmark

from piper_kernels._triton.targets import AcceleratorTarget
from piper_kernels.attention.piper_attention import _policy
from piper_kernels.attention.piper_attention import triton as piper_backend

_DEFAULT_SEQUENCES = (8192, 32768, 131072)
_DEFAULT_SEEDS = (0, 1, 2)
_MAXIMUM_SQNR_LOSS_DB = 0.5
_MAXIMUM_SATURATION_INCREASE = 0.001
type JSONValue = str | int | float | bool | list[JSONValue] | dict[str, JSONValue] | None


def _json_safe(value: object) -> JSONValue:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sqnr_margin(candidate: float, generic: float) -> float:
    if math.isinf(candidate) and math.isinf(generic):
        return 0.0
    return candidate - generic


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, nargs="+", default=_DEFAULT_SEQUENCES)
    parser.add_argument("--seeds", type=int, nargs="+", default=_DEFAULT_SEEDS)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--ablation-sequence", type=int, default=8192)
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--skip-regressions", action="store_true")
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--warmup-ms", type=int, default=50)
    parser.add_argument("--measurement-time-ms", type=int, default=300)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def _validate_args(arguments: argparse.Namespace) -> None:
    if not arguments.sequences or any(sequence <= 0 for sequence in arguments.sequences):
        raise SystemExit("sequences must contain positive values")
    if not arguments.seeds:
        raise SystemExit("at least one deterministic seed is required")
    if arguments.heads <= 0:
        raise SystemExit("heads must be positive")
    if arguments.ablation_sequence <= 0:
        raise SystemExit("ablation sequence must be positive")
    if arguments.warmup_ms < 0 or arguments.measurement_time_ms <= 0:
        raise SystemExit("warmup must be non-negative and measurement time must be positive")


def _generic_plan(
    production_plan: _policy.PiperAttentionExecutionPlan,
    *,
    packed_probability: bool,
) -> _policy.PiperAttentionExecutionPlan:
    """Recover the current-upstream generic SM89 path for a fair control."""
    is_causal = production_plan.optimize_causal_traversal
    return replace(
        production_plan,
        block_m=64 if is_causal else 128,
        use_sm89_d128_specialization=False,
        use_shared_value_scale=False,
        use_fused_kv_preprocessing=False,
        use_fp16_value_scale=False,
        derive_value_scale_multiplier=False,
        use_hybrid_fp32_fp16_numerator=False,
        use_strided_kv_mean_sample=False,
        split_pv_head_dim=not is_causal,
        scaled_fp16_numerator=False,
        num_stages=3 if is_causal else 1,
        optimize_causal_traversal=False,
        loop_num_stages=None if is_causal else 3,
        loop_licm=not is_causal,
        use_packed_probability_conversion=packed_probability,
        round_probability_codes=True,
    )


def _ablation_plans(
    production_plan: _policy.PiperAttentionExecutionPlan,
) -> tuple[tuple[str, _policy.PiperAttentionExecutionPlan], ...]:
    """Separate every performance/quality variable requested in issue #35."""
    return (
        ("generic-stock", _generic_plan(production_plan, packed_probability=False)),
        ("generic-packed", _generic_plan(production_plan, packed_probability=True)),
        (
            "dedicated-stock-per-key-fp32-acc-fp32-scale-unfused-round",
            replace(
                production_plan,
                use_packed_probability_conversion=False,
                scaled_fp16_numerator=False,
                use_fused_kv_preprocessing=False,
                use_fp16_value_scale=False,
                derive_value_scale_multiplier=False,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        (
            "dedicated-packed-per-key-fp32-acc-fp32-scale-unfused-round",
            replace(
                production_plan,
                scaled_fp16_numerator=False,
                use_fused_kv_preprocessing=False,
                use_fp16_value_scale=False,
                derive_value_scale_multiplier=False,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        (
            "dedicated-packed-per-key-fp32-acc-fp16-scale-unfused-round",
            replace(
                production_plan,
                scaled_fp16_numerator=False,
                use_fused_kv_preprocessing=False,
                use_fp16_value_scale=True,
                derive_value_scale_multiplier=False,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        (
            "dedicated-packed-per-key-split-fp16-fp16-scale-unfused-round",
            replace(
                production_plan,
                scaled_fp16_numerator=True,
                use_fused_kv_preprocessing=False,
                use_fp16_value_scale=True,
                derive_value_scale_multiplier=False,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        (
            "dedicated-packed-per-key-split-fp16-acc-fp16-scale-fused-derived-round",
            replace(
                production_plan,
                scaled_fp16_numerator=True,
                use_fp16_value_scale=True,
                derive_value_scale_multiplier=True,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        ("production", production_plan),
        (
            "dedicated-packed-shared-v64-production-acc-fp32-scale-fused-round",
            replace(
                production_plan,
                use_shared_value_scale=True,
                use_fp16_value_scale=False,
                derive_value_scale_multiplier=False,
                use_hybrid_fp32_fp16_numerator=False,
            ),
        ),
        (
            "dedicated-packed-per-key-production-acc-fp16-scale-fused-truncate",
            replace(production_plan, round_probability_codes=False),
        ),
    )


def _worst_head_sqnr(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return min(
        measure_quality(actual[:, head], reference[:, head]).sqnr_db
        for head in range(actual.shape[1])
    )


def _saturation_fractions(
    prepared: piper_backend._PreparedPiperAttention,
) -> dict[str, float]:
    tensors = {
        "query_int8": prepared.query,
        "key_int8": prepared.key,
        "value_int8": prepared.value,
    }
    return {
        name: measure_saturation(
            tensor,
            -127,
            127,
        ).fraction
        for name, tensor in tensors.items()
        if isinstance(tensor, torch.Tensor)
    }


def _run_plan(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    plan: _policy.PiperAttentionExecutionPlan,
    *,
    is_causal: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    query, key, value = inputs
    prepared = piper_backend._prepare_piper_attention(
        query,
        key,
        value,
        128**-0.5,
        is_causal,
        execution_plan=plan,
    )
    output = piper_backend._launch_piper_attention(prepared)
    torch.cuda.synchronize()
    return output, _saturation_fractions(prepared)


def _timing_record(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    candidate: str,
    plan: _policy.PiperAttentionExecutionPlan,
    sequence: int,
    is_causal: bool,
    warmup_ms: int,
    measurement_time_ms: int,
) -> dict[str, Any]:
    query, key, value = inputs
    prepared = piper_backend._prepare_piper_attention(
        query,
        key,
        value,
        128**-0.5,
        is_causal,
        execution_plan=plan,
    )
    piper_backend._launch_piper_attention(prepared)
    torch.cuda.synchronize()
    prepared_timing = triton_benchmark(
        lambda: piper_backend._launch_piper_attention(prepared),
        warmup_ms,
        measurement_time_ms,
    )
    complete_timing = synchronized_wall_benchmark(
        lambda: piper_backend._run_piper_attention(
            query,
            key,
            value,
            128**-0.5,
            is_causal,
            execution_plan=plan,
        ),
        warmup_ms,
        measurement_time_ms,
        synchronize=torch.cuda.synchronize,
    )
    return {
        "candidate": candidate,
        "sequence": sequence,
        "prepared_execution": prepared_timing.as_dict(),
        "complete_operator": complete_timing.as_dict(),
        "plan": plan.as_dict(),
    }


def _quality_record(
    *,
    kind: str,
    candidate: str,
    sequence: int,
    seed: int,
    output: torch.Tensor,
    generic_output: torch.Tensor,
    reference: torch.Tensor,
    saturation: Mapping[str, float],
    generic_saturation: Mapping[str, float],
    plan: _policy.PiperAttentionExecutionPlan,
) -> dict[str, Any]:
    quality = measure_quality(output, reference)
    generic_quality = measure_quality(generic_output, reference)
    sqnr_margin_db = _sqnr_margin(quality.sqnr_db, generic_quality.sqnr_db)
    worst_head_sqnr_db = _worst_head_sqnr(output, reference)
    generic_worst_head_sqnr_db = _worst_head_sqnr(generic_output, reference)
    worst_head_margin_db = _sqnr_margin(
        worst_head_sqnr_db,
        generic_worst_head_sqnr_db,
    )
    saturation_increase = max(
        (
            saturation.get(name, 0.0) - generic_saturation.get(name, 0.0)
            for name in set(saturation) | set(generic_saturation)
        ),
        default=0.0,
    )
    passed = (
        quality.nonfinite_mismatch_count == 0
        and quality.actual_nonfinite_count <= generic_quality.actual_nonfinite_count
        and sqnr_margin_db >= -_MAXIMUM_SQNR_LOSS_DB
        and worst_head_margin_db >= -_MAXIMUM_SQNR_LOSS_DB
        and saturation_increase <= _MAXIMUM_SATURATION_INCREASE
    )
    return {
        "kind": kind,
        "candidate": candidate,
        "sequence": sequence,
        "seed": seed,
        "passed": passed,
        "sqnr_db": quality.sqnr_db,
        "generic_sqnr_db": generic_quality.sqnr_db,
        "sqnr_margin_db": sqnr_margin_db,
        "worst_head_sqnr_db": worst_head_sqnr_db,
        "generic_worst_head_sqnr_db": generic_worst_head_sqnr_db,
        "worst_head_margin_db": worst_head_margin_db,
        "actual_nonfinite_count": quality.actual_nonfinite_count,
        "generic_nonfinite_count": generic_quality.actual_nonfinite_count,
        "saturation": dict(saturation),
        "generic_saturation": dict(generic_saturation),
        "maximum_saturation_increase": saturation_increase,
        "plan": plan.as_dict(),
    }


def _make_inputs(
    sequence: int,
    seed: int,
    heads: int,
    *,
    is_causal: bool = False,
    value_pattern: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = AttentionShape(1, heads, sequence, sequence, 128)
    config = AttentionConfig(
        dtype=torch.bfloat16,
        is_causal=is_causal,
        scale=128**-0.5,
        seed=seed,
    )
    query, key, value = make_attention_inputs(shape, config=config, device=torch.device("cuda"))
    if value_pattern == "biased":
        offset = torch.linspace(-8, 8, 128, device="cuda").reshape(1, 1, 1, 128)
        value = (offset + value.float() * 0.25).to(torch.bfloat16)
    elif value_pattern == "constant":
        value = value[:, :, :1].expand_as(value).contiguous()
    return query, key, value


def _evaluate_inputs(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    sequence: int,
    seed: int,
    kind: str,
    include_ablations: bool,
    is_causal: bool,
) -> list[dict[str, Any]]:
    query, key, _value = inputs
    production_plan = piper_backend._default_piper_attention_execution_plan(
        query,
        is_causal,
        key_length=key.shape[2],
    )
    if not production_plan.use_sm89_d128_specialization:
        raise RuntimeError("workload did not select the SM89 D128 specialization")
    generic_plan = _generic_plan(production_plan, packed_probability=False)
    generic_output, generic_saturation = _run_plan(
        inputs,
        generic_plan,
        is_causal=is_causal,
    )
    reference = run_sdpa(
        inputs,
        AttentionConfig(
            dtype=query.dtype,
            is_causal=is_causal,
            scale=128**-0.5,
            seed=seed,
        ),
    )
    candidates = (
        _ablation_plans(production_plan)
        if include_ablations
        else (("production", production_plan),)
    )
    records: list[dict[str, Any]] = []
    for name, plan in candidates:
        if name == "generic-stock":
            output = generic_output
            saturation = generic_saturation
        else:
            output, saturation = _run_plan(inputs, plan, is_causal=is_causal)
        records.append(
            _quality_record(
                kind=kind,
                candidate=name,
                sequence=sequence,
                seed=seed,
                output=output,
                generic_output=generic_output,
                reference=reference,
                saturation=saturation,
                generic_saturation=generic_saturation,
                plan=plan,
            )
        )
    return records


def _load_capture(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    capture = torch.load(path, map_location="cuda", weights_only=True)
    if not isinstance(capture, Mapping) or any(
        name not in capture for name in ("query", "key", "value")
    ):
        raise ValueError("capture must be a mapping containing query, key, and value tensors")
    inputs = tuple(capture[name] for name in ("query", "key", "value"))
    if not all(isinstance(tensor, torch.Tensor) for tensor in inputs):
        raise ValueError("capture query, key, and value entries must be tensors")
    return inputs  # type: ignore[return-value]


@torch.inference_mode()
def _main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_args(argv)
    _validate_args(arguments)
    if not torch.cuda.is_available():
        raise SystemExit("SM89 specialization validation requires an NVIDIA GPU")
    target = AcceleratorTarget.from_device(torch.device("cuda"))
    if not target.is_cuda_capability(8, 9):
        raise SystemExit("SM89 specialization validation requires compute capability 8.9")

    records: list[dict[str, Any]] = []
    timing_records: list[dict[str, Any]] = []
    for sequence in dict.fromkeys(arguments.sequences):
        for seed in dict.fromkeys(arguments.seeds):
            inputs = _make_inputs(
                sequence,
                seed,
                arguments.heads,
                is_causal=arguments.causal,
            )
            records.extend(
                _evaluate_inputs(
                    inputs,
                    sequence=sequence,
                    seed=seed,
                    kind="random",
                    include_ablations=(
                        not arguments.skip_ablations and sequence == arguments.ablation_sequence
                    ),
                    is_causal=arguments.causal,
                )
            )
            if seed == arguments.seeds[0] and not arguments.skip_timing:
                production_plan = piper_backend._default_piper_attention_execution_plan(
                    inputs[0],
                    arguments.causal,
                    key_length=inputs[1].shape[2],
                )
                timing_candidates = (
                    _ablation_plans(production_plan)
                    if not arguments.skip_ablations and sequence == arguments.ablation_sequence
                    else (
                        (
                            "generic-stock",
                            _generic_plan(production_plan, packed_probability=False),
                        ),
                        ("production", production_plan),
                    )
                )
                timing_records.extend(
                    _timing_record(
                        inputs,
                        candidate=name,
                        plan=plan,
                        sequence=sequence,
                        is_causal=arguments.causal,
                        warmup_ms=arguments.warmup_ms,
                        measurement_time_ms=arguments.measurement_time_ms,
                    )
                    for name, plan in timing_candidates
                )
            del inputs
            torch.cuda.empty_cache()

    if not arguments.skip_regressions:
        regression_sequence = min(arguments.sequences)
        regression_seed = arguments.seeds[0]
        for value_pattern in ("biased", "constant"):
            inputs = _make_inputs(
                regression_sequence,
                regression_seed,
                arguments.heads,
                is_causal=arguments.causal,
                value_pattern=value_pattern,
            )
            records.extend(
                _evaluate_inputs(
                    inputs,
                    sequence=regression_sequence,
                    seed=regression_seed,
                    kind=f"{value_pattern}-value",
                    include_ablations=False,
                    is_causal=arguments.causal,
                )
            )
            del inputs
            torch.cuda.empty_cache()

    if arguments.capture is not None:
        capture_inputs = _load_capture(arguments.capture)
        records.extend(
            _evaluate_inputs(
                capture_inputs,
                sequence=capture_inputs[0].shape[2],
                seed=-1,
                kind="real-model-capture",
                include_ablations=False,
                is_causal=arguments.causal,
            )
        )

    payload = {
        "schema_version": 1,
        "validation": "piper_sm89_d128_specialization",
        "is_causal": arguments.causal,
        "quality_gate": {
            "maximum_sqnr_loss_db": _MAXIMUM_SQNR_LOSS_DB,
            "maximum_saturation_fraction_increase": _MAXIMUM_SATURATION_INCREASE,
            "independent_per_seed_and_shape": True,
        },
        "environment": capture_environment(Path(__file__).resolve().parents[1]).as_dict(),
        "records": records,
        "timings": timing_records,
        "passed": all(
            record["passed"] for record in records if record["candidate"] == "production"
        ),
    }
    content = json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n"
    if arguments.json is not None:
        arguments.json.parent.mkdir(parents=True, exist_ok=True)
        arguments.json.write_text(content, encoding="utf-8")
    if arguments.quiet:
        production_records = [record for record in records if record["candidate"] == "production"]
        print(
            f"production quality gates: {sum(record['passed'] for record in production_records)}"
            f"/{len(production_records)} passed"
        )
    else:
        print(content, end="")
    if not payload["passed"]:
        raise SystemExit("production SM89 specialization failed its quality gate")


def main() -> None:
    """Run the SM89 quality gate and requested ablations."""
    _main()


if __name__ == "__main__":
    main()
