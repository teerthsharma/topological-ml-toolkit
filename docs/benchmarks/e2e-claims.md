# E2E Claim Benchmark

`benchmarks/e2e_claims.py` is the repository claim gate. It runs small,
deterministic workloads that verify the public behavior currently advertised by
the package.

The gate is intentionally strict about wording:

- active claims must have executable evidence;
- runtime-gated backends must expose explicit missing gates, not implied
  speedups;
- timed smoke results are recorded as timing records, not performance wins;
- optional stacks such as PyTorch, TensorFlow, and Triton must not load during
  `import topoml`.

## Active Claims

The current E2E gate verifies:

- known \(H_0\) cluster merges for a three-point cloud;
- an \(H_1\) square cycle that appears before diagonal/triangle filling;
- public deterministic benchmark dataset fixtures with expected topology
  metadata;
- time-delay embedding shape and first delay vector;
- fixed-width Betti-curve features from `PHFeaturizer`;
- `BettiCurve`, `PersistenceImage`, and topology signatures for point clouds,
  graphs, and activations;
- prototype finite topology, metric-cover, nerve, Mapper, sheaf residual,
  homotopy, strata, orbit, equivariance, Scott fixed-point, weak-convergence,
  sampled dynamics, braid-crossing, and mesh Euler diagnostics;
- TensorBundle interoperability, topology sample weights, and the
  dependency-light topology random forest baseline;
- optional sklearn `Pipeline` integration when scikit-learn is installed;
- Triton schedule construction with local and same-budget random ablations;
- graph-first gallery coverage for prototypes, persistence features, embedding
  graphs, backend runtime gates, topology training, and benchmark evidence;
- static self-contained GUI dashboard export;
- runnable tutorial scripts for point-cloud PH, optional sklearn integration,
  and dashboard export;
- backend metadata that separates active code from planned acceleration;
- backend source inventory for active C++, active hardware-gated Assembly,
  active optional CUDA, active optional Triton runtime work, and the CPU Triton
  schedule-construction benchmark;
- import safety for optional ML/GPU stacks;
- active optional PyTorch and TensorFlow adapter metadata;
- benchmark-smoke timing records for the Python reference path.
- CI-gated `ripser` and `GUDHI` parity on small Vietoris-Rips fixtures.

## What Is Not Claimed Yet

C++ is active for H0 barcode construction. ASM is active for CPUID/XCR0-gated
L2-squared dispatch only. CUDA is active for optional native pairwise-L2 and
threshold-edge preprocessing only. Triton is active for optional CUDA pairwise-L2
parity against `torch.cdist` only. PyTorch and TensorFlow are active optional
tensor/activation adapters. H1/H2 native reduction, topology-guided sparse
attention, and framework-native PH kernels remain gated until equivalence tests
and baseline benchmarks pass.

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
