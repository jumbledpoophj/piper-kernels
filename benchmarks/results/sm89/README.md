# SM89 attention tuning evidence

These records were captured on an RTX 4070 Ti SUPER and retain their complete
environment, revision, shape, configuration, timing, and quality metadata.

- [`piper/`](piper/README.md) contains the launch searches, native-versus-affine
  compiler reports, biased-value quality checks, final Piper measurements, and Piper
  versus canonical SageAttention2++ comparisons used for the SM89 dispatch policy.
- [`sage2pp/`](sage2pp/README.md) contains the clean-upstream comparison, final tuned
  matrix, compiler before/after records, two-session statistical non-inferiority study,
  and the rejected-experiment conclusions for pure-Triton SageAttention2++.

Only the final or decision-relevant evidence is retained here. Intermediate exploratory
passes, duplicate repeats, and thermally throttled measurements are intentionally not
versioned. Re-run a benchmark rather than treating these GPU-specific numbers as a
portable performance guarantee.
