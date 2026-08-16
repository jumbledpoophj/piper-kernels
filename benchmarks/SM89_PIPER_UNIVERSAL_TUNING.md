# SM89 Piper Attention: token-length-invariant tuning

## Outcome

The aligned SM89/D128 Piper Attention specialization now uses one production plan at every
supported token length. Sequence length does not select a kernel, precision, metadata format,
or launch schedule.

On an RTX 4070 Ti SUPER, the H16 end-to-end gap relative to the local pure-Triton
SageAttention2++ implementation is 1.74% to 3.80% for non-causal attention at 8K, 32K, and
128K. Piper is 1.89% to 2.90% faster for causal attention at the same lengths while retaining
37.31 dB or better SQNR. H48/128K is intentionally excluded from the benchmark matrix.

## Applicability

The dedicated path is selected when all of the following are true:

- the accelerator is exactly SM89;
- the head dimension is 128;
- query and key lengths are equal;
- query length is divisible by 128 and key length is divisible by 64.

Every length meeting those structural requirements receives the same mode-specific plan.
Unaligned, rectangular, non-D128, and non-SM89 inputs continue to use the generic path. This is
an alignment boundary required by the dedicated tiling, not a benchmark-anchor or token-count
threshold.

## Removed token-length cutovers

| Decision | Previous policy | Current policy |
|---|---|---|
| Dedicated-kernel admission | Non-causal from 8K; causal at 128K | Every aligned SM89/D128 self-attention length |
| Loop stages | Non-causal changed at 128K | Three stages at every length |
| Loop-invariant-code motion | Non-causal disabled at 128K | Enabled for every non-causal length |
| Numerator recurrence | Short non-causal used scaled FP16; long used hybrid | Hybrid FP32/FP16 at every length |
| V-scale storage | Short contexts used FP16 | FP32 per-key multipliers at every length |
| V-scale reconstruction | Short contexts reconstructed the multiplier | Loaded FP32 multiplier at every length |

The production source contains no 8K, 32K, or 128K dispatch decisions. Those values remain only
as measurement anchors in benchmark and validation tooling.

## Universal implementation choices

Both causal and non-causal modes use:

- 128-row query tiles, four warps, one outer pipeline stage, and three loop stages;
- pointer-based loads and split D128 PV tensor-core products;
- packed UINT8 probability conversion and rounded probability codes;
- fused Q/K/V preprocessing;
- exact full-sequence centering means wherever centering applies;
- per-key FP32 V-scale multipliers;
- one FP32 and one FP16 numerator half;
- magic-biased MMA accumulator conversion, avoiding scalar INT32-to-FP32 conversions.

The hybrid recurrence and metadata choices were selected as a family: hybrid accumulation
requires FP32 V-scale storage and a loaded multiplier. Execution-plan invariants reject
incompatible mixtures rather than allowing an accidental lower-quality configuration.

## Mode-specific choices

These choices depend on attention semantics, not sequence length:

| Mode | Centering | Traversal | Loop LICM |
|---|---|---|---|
| Non-causal | Exact K and V centering | Standard traversal | Enabled |
| Causal | Exact K centering; V remains uncentered | Longest CTAs first, mask-free prefix, masked boundary | Disabled |

Causal V remains uncentered so an earlier output cannot depend on future V rows through a
sequence-wide mean.

## Benchmark method

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (SM89) |
| Driver | 596.49 |
| Framework | PyTorch 2.13.0+cu130 |
| Triton | 3.7.1.post27 |
| Inputs | Batch 1, BF16, D128, square self-attention, seed 0 |
| Timing | 1,000 ms warmup; 10,000 ms measurement |
| Metric | Synchronized end-to-end operator median |
| Comparator | Repository-local pure-Triton SageAttention2++ |
| Quality reference | PyTorch scaled-dot-product attention |

The comparator is the local Triton SageAttention2++ provider, not the canonical CUDA extension.
The tables report one full-duration process run. They are suitable for checking the retained
performance target, but are not a new process-replicated statistical-significance study.

## H16 end-to-end results

Negative gaps mean Piper is faster.

### Non-causal

| Tokens | Piper (ms) | Local Triton SA2++ (ms) | Gap | Piper SQNR (dB) |
|---:|---:|---:|---:|---:|
| 8K | 2.886 | 2.833 | +1.87% | 36.80 |
| 32K | 42.316 | 41.591 | +1.74% | 36.61 |
| 128K | 651.327 | 627.472 | +3.80% | 36.39 |

### Causal

| Tokens | Piper (ms) | Local Triton SA2++ (ms) | Gap | Piper SQNR (dB) |
|---:|---:|---:|---:|---:|
| 8K | 1.742 | 1.794 | -2.90% | 37.61 |
| 32K | 22.163 | 22.754 | -2.60% | 37.38 |
| 128K | 328.208 | 334.522 | -1.89% | 37.31 |

## H48 validation results

| Mode | Tokens | Piper (ms) | Local Triton SA2++ (ms) | Gap | Piper SQNR (dB) |
|---|---:|---:|---:|---:|---:|
| Non-causal | 8K | 8.988 | 8.789 | +2.26% | 36.81 |
| Non-causal | 32K | 126.001 | 122.071 | +3.22% | 36.64 |
| Causal | 8K | 4.760 | 4.961 | -4.05% | 37.58 |
| Causal | 32K | 61.511 | 63.108 | -2.53% | 37.43 |

H48/128K was not run and should not be added to routine validation.

## Non-anchor continuity screen

A shorter 100 ms warmup and 500 ms measurement screen covered 2K, 12K, and 64K with H16. The
same execution plans were selected at every point. Non-causal gaps were -17.36%, +2.12%, and
+3.00%; causal gaps were -19.63%, -6.86%, and -6.10%, respectively. Piper SQNR remained between
36.61 and 37.74 dB. These cases guard against improvements that work only at the standard 8K,
32K, and 128K anchors.

## Reproduction

Run the H16 non-causal matrix:

```shell
uv run python benchmarks/benchmark_attention.py \
  --providers piper_attention sage_attention_2pp \
  --sequence 8192 32768 131072 \
  --batch-size 1 --heads 16 --head-dim 128 --dtype bfloat16 \
  --warmup-ms 1000 --measurement-time-ms 10000 \
  --json artifacts/sm89-piper-universal-h16-noncausal.json
```

Add `--causal` for the causal matrix. For H48, set `--heads 48` and limit `--sequence` to
`8192 32768`.

Run the production quality validator independently of performance ranking:

```shell
uv run python benchmarks/validate_piper_sm89_specialization.py \
  --sequences 8192 32768 131072 \
  --seeds 0 1 2 \
  --json artifacts/piper_sm89_validation.json
```

## Validation

- The full repository suite passes with 725 tests passed and 9 skipped.
- Policy tests compare the complete causal and non-causal plans at 128, 2K, 8K, 12K, 32K,
  64K, 128K, and 256K.
- GPU quality tests cover anchor and non-anchor lengths, multiple seeds, biased V, constant V,
  and causal future-value independence.
- Ruff and Git whitespace checks pass for the changed files.
