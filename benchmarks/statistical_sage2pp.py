"""Paired, counterbalanced non-inferiority benchmark for attention providers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "benchmarks"))
sys.path.insert(0, str(REPOSITORY / "src"))

from lib import attention_providers  # noqa: E402
from lib.attention import (  # noqa: E402
    AttentionConfig,
    AttentionShape,
    make_attention_inputs,
)
from lib.attention_providers import (  # noqa: E402
    CANONICAL_CUDA_SAGE_ATTENTION_2PP,
    PIPER_ATTENTION,
    SAGE_ATTENTION_2PP,
    AttentionProvider,
    make_attention_providers,
)
from lib.environment import capture_environment  # noqa: E402

from piper_kernels._triton.targets import AcceleratorTarget  # noqa: E402

attention_providers.CANONICAL_VERSION = importlib.metadata.version("sageattention")
attention_providers.CANONICAL_REVISION = (
    "woct0rdho/SageAttention@v2.2.0-windows.post5:3b90c0ec112b6b222db68fa160470dd492106ec0"
)

PROVIDERS = (SAGE_ATTENTION_2PP, CANONICAL_CUDA_SAGE_ATTENTION_2PP)
SHORT_NAMES = {
    SAGE_ATTENTION_2PP: "T",
    CANONICAL_CUDA_SAGE_ATTENTION_2PP: "C",
}
PATTERNS = {
    "TCCT": (
        SAGE_ATTENTION_2PP,
        CANONICAL_CUDA_SAGE_ATTENTION_2PP,
        CANONICAL_CUDA_SAGE_ATTENTION_2PP,
        SAGE_ATTENTION_2PP,
    ),
    "CTTC": (
        CANONICAL_CUDA_SAGE_ATTENTION_2PP,
        SAGE_ATTENTION_2PP,
        SAGE_ATTENTION_2PP,
        CANONICAL_CUDA_SAGE_ATTENTION_2PP,
    ),
}


def _configure_comparison(comparison: str) -> None:
    """Configure candidate/reference providers while retaining the proven design."""
    global PROVIDERS, SHORT_NAMES, PATTERNS  # noqa: PLW0603 - CLI-selected design
    if comparison == "triton-vs-canonical":
        candidate, reference = SAGE_ATTENTION_2PP, CANONICAL_CUDA_SAGE_ATTENTION_2PP
        candidate_short, reference_short = "T", "C"
    else:
        candidate, reference = PIPER_ATTENTION, SAGE_ATTENTION_2PP
        candidate_short, reference_short = "P", "S"
    PROVIDERS = (candidate, reference)
    SHORT_NAMES = {
        candidate: candidate_short,
        reference: reference_short,
    }
    PATTERNS = {
        "ABBA": (candidate, reference, reference, candidate),
        "BAAB": (reference, candidate, candidate, reference),
    }


def _telemetry() -> dict[str, str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.current.sm,clocks.current.memory,"
        "power.draw,pstate,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = (
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except FileNotFoundError, subprocess.CalledProcessError, IndexError:
        return {
            "temperature_c": None,
            "sm_clock_mhz": None,
            "memory_clock_mhz": None,
            "power_w": None,
            "pstate": None,
            "utilization_pct": None,
        }
    values = [value.strip() for value in output.split(",")]
    keys = (
        "temperature_c",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "power_w",
        "pstate",
        "utilization_pct",
    )
    return dict(zip(keys, values, strict=True))


def _time_provider(
    provider: AttentionProvider,
    prepared: object,
    iterations: int,
    phase: str,
) -> float:
    if phase == "operator_end_to_end":
        torch.cuda.synchronize()
        started_wall = time.perf_counter()
        output = None
        for _ in range(iterations):
            output = provider.run(provider.prepare())
        torch.cuda.synchronize()
        if output is None:
            raise AssertionError("iterations must be positive")
        return (time.perf_counter() - started_wall) * 1000.0 / iterations
    torch.cuda.synchronize()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()
    output = None
    for _ in range(iterations):
        output = provider.run(prepared)
    finished.record()
    finished.synchronize()
    if output is None:
        raise AssertionError("iterations must be positive")
    return float(started.elapsed_time(finished)) / iterations


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability, method="linear"))


def _bootstrap_distributions(
    log_ratios: np.ndarray,
    patterns: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return stratified, three-block-cluster, and order-effect bootstraps."""
    rng = np.random.default_rng(seed)
    pattern_a_name, pattern_b_name = PATTERNS
    pattern_a = log_ratios[patterns == pattern_a_name]
    pattern_b = log_ratios[patterns == pattern_b_name]
    stratified = np.empty(replicates, dtype=np.float64)
    order_effect = np.empty(replicates, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        sample_a = pattern_a[rng.integers(0, len(pattern_a), size=(count, len(pattern_a)))].mean(
            axis=1
        )
        sample_b = pattern_b[rng.integers(0, len(pattern_b), size=(count, len(pattern_b)))].mean(
            axis=1
        )
        stratified[start : start + count] = (
            sample_a * len(pattern_a) + sample_b * len(pattern_b)
        ) / len(log_ratios)
        order_effect[start : start + count] = sample_a - sample_b

    cluster_size = 3
    cluster_count = len(log_ratios) // cluster_size
    if cluster_count * cluster_size != len(log_ratios):
        raise ValueError("block count must be divisible by the three-block cluster size")
    clusters = log_ratios.reshape(cluster_count, cluster_size).mean(axis=1)
    clustered = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        samples = clusters[rng.integers(0, cluster_count, size=(count, cluster_count))]
        clustered[start : start + count] = samples.mean(axis=1)
    return stratified, clustered, order_effect


def _paired_randomization_p_value(
    log_ratios: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(log_ratios.mean()))
    exceedances = 0
    chunk_size = 10_000
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        signs = rng.integers(0, 2, size=(count, len(log_ratios)), dtype=np.int8)
        signs = signs * 2 - 1
        permuted = np.abs((signs * log_ratios).mean(axis=1))
        exceedances += int(np.count_nonzero(permuted >= observed))
    return (exceedances + 1) / (replicates + 1)


def _lag_one_correlation(values: np.ndarray) -> float | None:
    if len(values) < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _analyze(
    blocks: list[dict[str, object]],
    *,
    seed: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
    comparisons: int,
    margin_ratio: float,
) -> dict[str, object]:
    log_ratios = np.asarray([block["log_ratio"] for block in blocks], dtype=np.float64)
    patterns = np.asarray([block["pattern"] for block in blocks])
    stratified, clustered, order_effect = _bootstrap_distributions(
        log_ratios,
        patterns,
        seed=seed,
        replicates=bootstrap_replicates,
    )
    point = math.exp(float(log_ratios.mean()))
    stratified_ratio = np.exp(stratified)
    clustered_ratio = np.exp(clustered)
    individual_upper_probability = 0.95
    familywise_upper_probability = 1.0 - 0.05 / comparisons
    stratified_ci = (
        _quantile(stratified_ratio, 0.025),
        _quantile(stratified_ratio, 0.975),
    )
    cluster_ci = (
        _quantile(clustered_ratio, 0.025),
        _quantile(clustered_ratio, 0.975),
    )
    conservative_ci = (
        min(stratified_ci[0], cluster_ci[0]),
        max(stratified_ci[1], cluster_ci[1]),
    )
    individual_upper = max(
        _quantile(stratified_ratio, individual_upper_probability),
        _quantile(clustered_ratio, individual_upper_probability),
    )
    familywise_upper = max(
        _quantile(stratified_ratio, familywise_upper_probability),
        _quantile(clustered_ratio, familywise_upper_probability),
    )
    order_ratio = np.exp(order_effect)
    order_ci = (_quantile(order_ratio, 0.025), _quantile(order_ratio, 0.975))
    ratio_values = np.exp(log_ratios)
    return {
        "estimand": "geometric_mean_of_paired_block_latency_ratios",
        "candidate_over_reference_ratio": point,
        "gap_percent": (point - 1.0) * 100.0,
        "ratio_median": float(np.median(ratio_values)),
        "ratio_p20": _quantile(ratio_values, 0.2),
        "ratio_p80": _quantile(ratio_values, 0.8),
        "stratified_bootstrap_two_sided_95_ci": list(stratified_ci),
        "three_block_cluster_bootstrap_two_sided_95_ci": list(cluster_ci),
        "conservative_two_sided_95_ci": list(conservative_ci),
        "individual_one_sided_95_upper": individual_upper,
        "familywise_bonferroni_one_sided_upper": familywise_upper,
        "familywise_upper_confidence_level": familywise_upper_probability,
        "noninferiority_margin_ratio": margin_ratio,
        "within_margin_noninferior_individual_95": individual_upper < margin_ratio,
        "within_margin_noninferior_familywise_95": familywise_upper < margin_ratio,
        "equality_difference_significant_95": not (conservative_ci[0] <= 1.0 <= conservative_ci[1]),
        "paired_randomization_p_value_equality_unadjusted": _paired_randomization_p_value(
            log_ratios,
            seed=seed + 1,
            replicates=permutation_replicates,
        ),
        "pattern_abba_ratio": math.exp(float(log_ratios[patterns == "ABBA"].mean())),
        "pattern_baab_ratio": math.exp(float(log_ratios[patterns == "BAAB"].mean())),
        "order_effect_abba_over_baab_ratio_95_ci": list(order_ci),
        "order_effect_detected_95": not (order_ci[0] <= 1.0 <= order_ci[1]),
        "lag_one_correlation_log_ratio": _lag_one_correlation(log_ratios),
        "bootstrap_replicates": bootstrap_replicates,
        "permutation_replicates": permutation_replicates,
    }


def _holm_adjust(results: list[dict[str, object]]) -> None:
    indexed = sorted(
        enumerate(results),
        key=lambda item: item[1]["analysis"]["paired_randomization_p_value_equality_unadjusted"],
    )
    running = 0.0
    total = len(indexed)
    for rank, (original_index, result) in enumerate(indexed):
        raw = float(result["analysis"]["paired_randomization_p_value_equality_unadjusted"])
        running = max(running, min(1.0, raw * (total - rank)))
        results[original_index]["analysis"]["paired_randomization_p_value_equality_holm"] = running


def _run_shape(  # noqa: PLR0913 - explicit statistical design parameters
    *,
    sequence: int,
    is_causal: bool,
    blocks: int,
    target_segment_ms: float,
    warmup_seconds: float,
    seed: int,
    bootstrap_replicates: int,
    permutation_replicates: int,
    comparisons: int,
    phase: str,
    margin_ratio: float,
) -> dict[str, object]:
    device = torch.device("cuda")
    target = AcceleratorTarget.from_device(device)
    shape = AttentionShape(
        batch_size=1,
        num_query_heads=8,
        query_length=sequence,
        key_value_length=sequence,
        head_dim=128,
    )
    config = AttentionConfig(dtype="bfloat16", is_causal=is_causal)
    generator = torch.Generator(device=device).manual_seed(seed)
    inputs = make_attention_inputs(
        shape,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    providers = make_attention_providers(
        inputs,
        provider_names=PROVIDERS,
        config=config,
        target=target,
    )
    prepared = {name: providers[name].prepare() for name in PROVIDERS}

    for name in PROVIDERS:
        providers[name].run(prepared[name])
        torch.cuda.synchronize()

    calibration: dict[str, list[float]] = {}
    for name in PROVIDERS:
        calibration[name] = [
            _time_provider(providers[name], prepared[name], 1, phase) for _ in range(3)
        ]
    slowest = max(statistics.median(values) for values in calibration.values())
    iterations = max(1, min(512, round(target_segment_ms / slowest)))

    warmup_started = time.perf_counter()
    warmup_index = 0
    while time.perf_counter() - warmup_started < warmup_seconds:
        name = PROVIDERS[warmup_index % len(PROVIDERS)]
        _time_provider(providers[name], prepared[name], iterations, phase)
        warmup_index += 1

    rng = np.random.default_rng(seed + 1000)
    pattern_a, pattern_b = PATTERNS
    pattern_names = [pattern_a] * (blocks // 2) + [pattern_b] * (blocks // 2)
    rng.shuffle(pattern_names)
    observations: list[dict[str, object]] = []
    print(
        f"start N={sequence} causal={is_causal} iterations={iterations} "
        f"calibration={calibration} telemetry={_telemetry()}",
        flush=True,
    )
    for block_index, pattern_name in enumerate(pattern_names, start=1):
        segments: list[dict[str, object]] = []
        for provider_name in PATTERNS[pattern_name]:
            latency = _time_provider(
                providers[provider_name],
                prepared[provider_name],
                iterations,
                phase,
            )
            segments.append(
                {
                    "provider": provider_name,
                    "short_name": SHORT_NAMES[provider_name],
                    "latency_ms": latency,
                }
            )
        candidate_short = SHORT_NAMES[PROVIDERS[0]]
        reference_short = SHORT_NAMES[PROVIDERS[1]]
        candidate_values = [
            float(segment["latency_ms"])
            for segment in segments
            if segment["short_name"] == candidate_short
        ]
        reference_values = [
            float(segment["latency_ms"])
            for segment in segments
            if segment["short_name"] == reference_short
        ]
        candidate_latency = statistics.fmean(candidate_values)
        reference_latency = statistics.fmean(reference_values)
        ratio = candidate_latency / reference_latency
        observation = {
            "block": block_index,
            "pattern": pattern_name,
            "segments": segments,
            "candidate_latency_ms": candidate_latency,
            "reference_latency_ms": reference_latency,
            "ratio": ratio,
            "log_ratio": math.log(ratio),
            "telemetry_after": _telemetry(),
        }
        observations.append(observation)
        if block_index == 1 or block_index % 5 == 0 or block_index == blocks:
            running_ratio = math.exp(
                statistics.fmean(float(item["log_ratio"]) for item in observations)
            )
            print(
                f"progress N={sequence} causal={is_causal} block={block_index}/{blocks} "
                f"running_gap_pct={(running_ratio - 1.0) * 100.0:+.3f} "
                f"telemetry={observation['telemetry_after']}",
                flush=True,
            )

    analysis = _analyze(
        observations,
        seed=seed + 2000,
        bootstrap_replicates=bootstrap_replicates,
        permutation_replicates=permutation_replicates,
        comparisons=comparisons,
        margin_ratio=margin_ratio,
    )
    del providers, prepared, inputs
    torch.cuda.empty_cache()
    return {
        "shape": shape.as_dict(),
        "configuration": config.as_dict(),
        "seed": seed,
        "calibration_single_call_ms": calibration,
        "iterations_per_segment": iterations,
        "warmup_seconds": warmup_seconds,
        "phase": phase,
        "blocks": observations,
        "analysis": analysis,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--comparison",
        choices=("triton-vs-canonical", "piper-vs-triton"),
        default="triton-vs-canonical",
    )
    parser.add_argument(
        "--phase",
        choices=("prepared_execution", "operator_end_to_end"),
        default="prepared_execution",
    )
    parser.add_argument("--margin-percent", type=float, default=5.0)
    parser.add_argument(
        "--sequence",
        type=int,
        nargs="+",
        default=[8192, 32768, 131072],
    )
    parser.add_argument(
        "--causal-mode",
        choices=("both", "noncausal", "causal"),
        default="both",
    )
    parser.add_argument("--familywise-comparisons", type=int, default=6)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--target-segment-ms", type=float, default=200.0)
    parser.add_argument("--warmup-seconds", type=float, default=4.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=250_000)
    parser.add_argument("--permutation-replicates", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--reverse-jobs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_comparison(args.comparison)
    if args.blocks < 12 or args.blocks % 6:
        raise SystemExit("--blocks must be at least 12 and divisible by 6")
    if args.margin_percent <= 0:
        raise SystemExit("--margin-percent must be positive")
    if any(sequence <= 0 for sequence in args.sequence):
        raise SystemExit("--sequence values must be positive")
    if args.familywise_comparisons <= 0:
        raise SystemExit("--familywise-comparisons must be positive")
    margin_ratio = 1.0 + args.margin_percent / 100.0
    environment = capture_environment(REPOSITORY).as_dict()
    causal_values = (
        (False, True)
        if args.causal_mode == "both"
        else (args.causal_mode == "causal",)
    )
    jobs = [
        (sequence, causal)
        for causal in causal_values
        for sequence in args.sequence
    ]
    if args.reverse_jobs:
        jobs.reverse()
    results: list[dict[str, object]] = []
    for index, (sequence, causal) in enumerate(jobs):
        results.append(
            _run_shape(
                sequence=sequence,
                is_causal=causal,
                blocks=args.blocks,
                target_segment_ms=args.target_segment_ms,
                warmup_seconds=args.warmup_seconds,
                seed=args.seed + index,
                bootstrap_replicates=args.bootstrap_replicates,
                permutation_replicates=args.permutation_replicates,
                comparisons=args.familywise_comparisons,
                phase=args.phase,
                margin_ratio=margin_ratio,
            )
        )
    _holm_adjust(results)
    pattern_design = {
        name: [SHORT_NAMES[item] for item in pattern] for name, pattern in PATTERNS.items()
    }
    document = {
        "schema_version": 1,
        "benchmark": "attention_paired_noninferiority",
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "environment": environment,
        "comparison": args.comparison,
        "design": {
            "providers": list(PROVIDERS),
            "estimand": "geometric mean of paired block latency ratios",
            "patterns": pattern_design,
            "blocks_per_shape": args.blocks,
            "patterns_per_shape": dict.fromkeys(PATTERNS, args.blocks // 2),
            "target_segment_ms": args.target_segment_ms,
            "warmup_seconds": args.warmup_seconds,
            "phase": args.phase,
            "candidate": PROVIDERS[0],
            "reference": PROVIDERS[1],
            "noninferiority_margin_ratio": margin_ratio,
            "individual_alpha_one_sided": 0.05,
            "familywise_alpha_one_sided": 0.05,
            "multiplicity_correction": "Bonferroni across six shapes",
            "familywise_comparisons": args.familywise_comparisons,
            "bootstrap": "pattern-stratified and consecutive three-block cluster percentile",
            "equality_test": "paired sign-flip randomization with Holm correction",
            "clock": (
                "synchronized wall clock over repeated prepare+run calls"
                if args.phase == "operator_end_to_end"
                else "CUDA device events over repeated prepared executions"
            ),
            "job_order": [{"sequence": sequence, "is_causal": causal} for sequence, causal in jobs],
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    for result in results:
        shape = result["shape"]
        config = result["configuration"]
        analysis = result["analysis"]
        print(
            f"result N={shape['query_length']} causal={config['is_causal']} "
            f"gap_pct={analysis['gap_percent']:+.3f} "
            f"ci95={analysis['conservative_two_sided_95_ci']} "
            f"familywise_upper={analysis['familywise_bonferroni_one_sided_upper']:.6f} "
            f"noninferior={analysis['within_margin_noninferior_familywise_95']} "
            f"holm_p={analysis['paired_randomization_p_value_equality_holm']:.6g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
