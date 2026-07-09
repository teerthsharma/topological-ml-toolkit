# E2E Claim Benchmark

`benchmarks/e2e_claims.py` is the repository claim gate. It runs small,
deterministic workloads that verify the public behavior currently advertised by
the package.

The gate is intentionally strict about wording:

- active claims must have executable evidence;
- planned backends must be discoverable as API contracts, not implied speedups;
- timed smoke results are recorded as timing records, not performance wins;
- optional stacks such as PyTorch, TensorFlow, and Triton must not load during
  `import topoml`.

## Active Claims

The current E2E gate verifies:

- known \(H_0\) cluster merges for a three-point cloud;
- an \(H_1\) square cycle that appears before diagonal/triangle filling;
- time-delay embedding shape and first delay vector;
- fixed-width Betti-curve features from `PHFeaturizer`;
- `BettiCurve`, `PersistenceImage`, and topology signatures for point clouds,
  graphs, and activations;
- prototype metric-cover, nerve, Mapper, and sheaf residual diagnostics;
- static self-contained GUI dashboard export;
- backend metadata that separates active code from planned acceleration;
- planned backend source inventory for CUDA, Assembly, C++, and Triton;
- import safety for optional ML/GPU stacks;
- benchmark-smoke timing records for the Python reference path.
- CI-gated `ripser` and `GUDHI` parity on small Vietoris-Rips fixtures.

## What Is Not Claimed Yet

The gate does not claim that C++, ASM AVX-512, Triton, PyTorch, or TensorFlow
backends are implemented. Those backends are roadmap/API-level surfaces until
their equivalence tests, fallback behavior, and baseline benchmarks pass.

The gate also does not claim a speedup over `ripser`, GUDHI, sklearn, dense
SDPA, FlashAttention, or any framework kernel. Those comparisons belong in
backend-specific benchmark suites once those backends exist.

## Local Command

```powershell
python benchmarks/e2e_claims.py --json-out artifacts/e2e-claims.json --md-out artifacts/e2e-claims.md
```

CI uploads the generated JSON and Markdown as benchmark artifacts.

The external TDA baseline artifact is produced separately by
`benchmarks/benchmark_tda_baselines.py` so parity against established libraries
is not confused with the internal active-claim smoke report.
