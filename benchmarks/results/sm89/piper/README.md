# SM89 Piper Attention reports

These JSON files are the machine-readable evidence for the SM89 dispatch policy. They
were captured on an RTX 4070 Ti SUPER with the environment serialized inside every
record.

- `tuning-*.json`: complete offline launch candidate searches for D64/D128,
  causal/non-causal N=8192, and non-causal H1/N131072.
- `benchmark-h8-*.json`: centered native, uncentered native, centered affine, and
  PyTorch SDPA comparisons across short and long B1/H8 shapes.
- `benchmark-video-*.json`: H1/N32768 and H1/N131072 long-context comparisons.
- `benchmark-biased-*.json`: centered versus uncentered quality with deterministic
  per-feature V bias amplitude 8.
- `compiler-*.json`: Triton resource and PTX instruction reports for native and affine
  N=8192 specializations. SASS is null because `nvdisasm` was unavailable.
- `comparison-sage2pp/`: final Piper versus canonical SageAttention2++ comparisons at
  B1/H8/D128 and 8K, 32K, and 128K in causal and non-causal modes.

All records use the benchmark schema documented in `benchmarks/README.md`. Re-run a
search or measurement rather than treating these GPU-specific numbers as portable
performance guarantees.
