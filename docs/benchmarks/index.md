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

## Native C++ Distance Benchmark

Ubuntu CI runs a native C++ preprocessing smoke benchmark:

```powershell
python benchmarks/benchmark_native_distance.py --out artifacts/native-distance.json --points 8 16
```

This compiles the portable C++ C ABI, loads it through `ctypes`, compares
pairwise distances and threshold edges against NumPy, and writes timing evidence.
It is a preprocessing correctness/timing smoke, not a persistent-homology
acceleration claim.
