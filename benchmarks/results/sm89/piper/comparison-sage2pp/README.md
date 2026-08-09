# SM89 Piper versus SageAttention2++ comparisons

This directory separates two comparison lifecycles captured on an RTX 4070 Ti SUPER:

- `sm89-piper-vs-canonical-sa2pp-*.json` contains the earlier per-shape point
  comparisons against the revision-pinned official CUDA SageAttention2++ package.
- `sm89-piper-vs-triton-sa2pp-statistical-complete-torch213.json` is the curated
  final comparison against this repository's local Triton SageAttention2++ provider.
  It measures complete-operator wall time, including preprocessing and allocations.

The statistical summary covers BF16 B1/H8/D128 self-attention at 8K, 32K, and 128K
in causal and non-causal modes. Each shape uses 30 paired ABBA/BAAB blocks, a 3%
non-inferiority margin, 250,000 bootstrap replicates, and a familywise Bonferroni
upper bound across all six shapes. Negative gaps mean Piper is faster.

The summary intentionally omits the raw per-block trace. Re-run
`benchmarks/statistical_sage2pp.py` when auditing the bootstrap or testing another
GPU, software version, tensor shape, or provider revision.
