# SM89 pure-Triton SageAttention2++ findings

## Retained implementation

The final implementation is based on upstream revision `962ca4c` and retains three
measured changes:

1. A packed inline-PTX FP32-to-E4M3 conversion uses two native
   `cvt.rn.satfinite.e4m3x2.f32` instructions per four values. It replaces the stock
   Triton software conversion for probability and V quantization on supported NVIDIA
   FP8 targets.
2. SM89 D128 causal calls at N>=8192 use 128 query rows, four warps, two launch stages,
   and reverse query-block order so the longest causal workgroups launch first.
3. SM89 D128 non-causal calls at N>=8192 enable loop-invariant-code motion and a
   three-stage `tl.range` pipeline while retaining 128 query rows and 64-key tiles.

The architecture and shape-specific choices live in `_Sage2ppExecutionPlan`; no runtime
autotuning is added to the user hot path.

## Final paired matrix

BF16 B1/H8/D128 complete GPU-operator medians on the RTX 4070 Ti SUPER:

| N | mode | Triton | canonical CUDA | gap | SQNR vs SDPA |
|---:|:---|---:|---:|---:|---:|
| 8,192 | non-causal | 1.320 ms | 1.326 ms | -0.5% | 28.42 dB |
| 8,192 | causal | 0.914 ms | 0.924 ms | -1.1% | 29.02 dB |
| 32,768 | non-causal | 20.241 ms | 19.987 ms | +1.3% | 28.28 dB |
| 32,768 | causal | 11.060 ms | 11.130 ms | -0.6% | 28.92 dB |
| 131,072 | non-causal | 310.977 ms | 302.304 ms | +2.9% | 28.25 dB |
| 131,072 | causal | 164.773 ms | 157.340 ms | +4.7% | 28.79 dB |

`final-noncausal.json` and `final-causal.json` contain the complete records. Negative
gaps mean Triton is faster.

Relative to clean upstream, retained tuning reduced hot latency by 19-23% non-causal
and 30-40% causal across these lengths.

## Compiler evidence

At 32K non-causal, the final attention specialization reports 2,500 static PTX
instructions, 255 registers per thread, 10 spills, 49,664 shared bytes per workgroup,
and two resident workgroups per SM. The pre-LICM specialization had the same static PTX
count and residency with no spills, but measured 20.969 ms versus 20.088 ms for the
cleaned final loop. The win is scheduling and overlap rather than a smaller static body.

The corresponding machine-readable records are
`compiler-baseline-n32768-noncausal.json` and
`compiler-final-n32768-noncausal.json`. SASS is absent because `nvdisasm` was not
installed on the benchmark host.

## Statistical non-inferiority study

The 5% non-inferiority claim was tested twice in independent Python processes with
opposite shape order and different data/randomization seeds. Each session used 30
counterbalanced paired blocks per shape, split equally between `T-C-C-T` and `C-T-T-C`.
The analysis used pattern-stratified and consecutive three-block cluster bootstrap
resampling, plus a Bonferroni one-sided correction across all six shapes.

| N | mode | equal-session gap | worst simultaneous upper bound | within 5% twice |
|---:|:---|---:|---:|:---:|
| 8,192 | non-causal | -1.962% | -0.510% | yes |
| 8,192 | causal | -1.689% | -1.543% | yes |
| 32,768 | non-causal | +0.564% | +4.033% | yes |
| 32,768 | causal | -1.124% | -0.930% | yes |
| 131,072 | non-causal | +2.807% | +3.037% | yes |
| 131,072 | causal | +4.314% | +4.690% | yes |

Both sessions independently keep every simultaneous upper bound below the predeclared
+5% margin. The 32K non-causal difference from equal speed was not significant in the
noisier replication, but its non-inferiority result remained significant. The raw 360
paired blocks, telemetry, bootstrap results, and multiplicity-corrected tests are in
`statistical-session-1.json` and `statistical-session-2.json`. Reproduce the study with
`benchmarks/statistical_sage2pp.py`.

```powershell
python benchmarks/statistical_sage2pp.py `
  --output artifacts/sage2pp-paired-noninferiority.json `
  --blocks 30 --target-segment-ms 200 --warmup-seconds 4
```

## Rejected experiments

- Canonical-style approximate reciprocal was neutral at 8K/32K and 0.7% slower at
  128K non-causal, so exact division remains.
- Explicit loop stages one and two regressed. Stage three combined with LICM won;
  stage four, unroll factor two, accumulator multi-buffer suppression, register caps,
  loop flattening, and disabled FP fusion were neutral or slower.
- With LICM active at 32K non-causal, 64- and 32-row query tiles were 48% and 83%
  slower; two and eight warps were 187% and 28% slower.
- Eight warps lost on causal mode, and one, three, or four causal launch stages lost to
  two stages for the long SM89 D128 schedule.
- Approximate reciprocal and other rejected controls are not left in the production
  execution plan.

## Scope

These results apply to the recorded RTX 4070 Ti SUPER, software stack, and shapes. The
packed FP8 conversion is valid on SM89 and newer PTX targets, including SM120, but its
performance gain has only been measured here on physical SM89 hardware. The launch and
loop policies are deliberately gated to the measured SM89 D128 regimes.
