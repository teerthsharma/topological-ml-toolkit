# Benchmark Policy

Benchmarks are generated from scripts. Tables in docs are summaries, not hand
typed truth.

## Workloads

- known clusters with known H0;
- circles with known H1;
- spheres and tori when H2 is enabled;
- Swiss roll embeddings;
- model activation point clouds;
- Q/K/V captures for sparse attention.

## Output Contract

Full benchmark artifacts should record:

- backend id;
- dtype;
- CPU/GPU/device details;
- seed;
- build time;
- execution time;
- peak memory when measurable;
- warning list;
- correctness metric.

The current E2E claim artifact is smaller because it is a correctness and API
claim gate, not a full performance study. It records deterministic evidence for
active public behavior and marks timing rows as smoke records rather than
speedup claims.

## CUDA Tensor Benchmark

GPU machines can run a manual CUDA tensor benchmark:

```powershell
python benchmarks/benchmark_cuda_tensors.py --out artifacts/cuda-tensor-topology.json
```

This benchmark uses real CUDA tensors and CUDA events. It runs a PyTorch
projection and `torch.cdist` on the GPU, then feeds the selected activation cloud
into the current topology stack. The persistent homology reduction is still CPU
reference code, so the artifact separates `cuda_ms` from `cpu_topology_ms`.

The script exits instead of producing fake data when CUDA is unavailable.

## Native C++ H0 Benchmark

Ubuntu CI runs a native C++ preprocessing and H0 barcode smoke benchmark:

```powershell
python benchmarks/benchmark_native_distance.py --out artifacts/native-distance.json --points 8 16
```

This compiles the portable C++ C ABI, loads it through `ctypes`, compares
pairwise distances and threshold edges against NumPy, compares H0 barcode deaths
against the Python reference, and writes timing evidence. It is an H0 native
correctness/timing smoke, not an H1/H2 acceleration claim.

## ASM Distance Benchmark

Ubuntu CI also runs a Linux x86-64 assembly smoke benchmark:

```powershell
python benchmarks/benchmark_asm_distance.py --out artifacts/asm-distance.json --points 8 16 --dims 8
```

This assembles the CPUID/XCR0 probes, scalar L2 hot path, and AVX-512 L2 hot
path, loads them through `ctypes`, compares dispatched output against NumPy, and
records which path actually ran. It is a dispatch and correctness gate, not a
PH acceleration or speedup claim.

## ML Adapter Integration Benchmark

The `ml integration / cpu frameworks` CI job runs real CPU PyTorch and
TensorFlow adapter checks:

```powershell
python benchmarks/benchmark_ml_adapters.py --out artifacts/ml-adapters.json
```

The artifact records framework versions, dtype/device metadata, topology
signature values, TensorFlow graph-mode parity, and PyTorch compile-safe capture
evidence. It is adapter integration evidence, not a model-quality or accelerated
persistent-homology claim.

## External TDA Baseline Benchmark

CI installs real `ripser` and `GUDHI` wheels and runs:

```powershell
python -m pytest python/tests/test_tda_baseline_parity.py -q
python benchmarks/benchmark_tda_baselines.py --out artifacts/tda-baselines.json
```

The parity fixtures are deliberately small and exact:

- three collinear points check finite \(H_0\) deaths at \(0.2\) and \(4.8\);
- a unit square checks that \(H_1\) is alive at radius \(1.1\) and killed by
  radius \(1.5\).

This gate proves the current Python reference path agrees with established TDA
libraries on the active Vietoris-Rips convention. It does not claim runtime
leadership, large-scale barcode equivalence, or C++/ASM/GPU acceleration.
