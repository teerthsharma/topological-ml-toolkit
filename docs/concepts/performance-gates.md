# Performance Gates

Speed is not a vibe. Speed is a measured claim.

## Required Baselines

Every benchmark must declare:

- fastest dense baseline available;
- safe Rust backend;
- Python wrapper overhead;
- `ripser` or `GUDHI` when installed;
- same-budget random selector for sparse methods;
- same-budget locality selector for sparse methods.

## Sparse Attention Gate

For topology-derived sparse attention:

\[
\text{relative L2} = \frac{\|y_{dense} - y_{sparse}\|_2}{\|y_{dense}\|_2}
\]

Promotion requires:

- relative L2 at or below declared tolerance;
- cosine similarity at or above declared tolerance;
- schedule build time reported separately;
- kernel execution time reported separately;
- comparison against dense SDPA or FlashAttention, not only slow CSR.

## Claim Policy

A README speed claim must point to raw benchmark output. No artifact, no claim.

