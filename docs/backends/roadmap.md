# Backend Roadmap

The backend ladder is deliberate.

## Safe Rust

Default backend. Bounded exact PH through small dimensions. This is the truth
source for tests.

## C++

Portable native path for users who cannot use Rust extensions directly.

Current source: `backends/cpp/topoml_native.cpp` defines a C ABI for pairwise
distance and threshold-edge preprocessing.

Gate:

- ABI documented;
- barcode equivalence against Rust;
- sanitizer-clean test run.

## ASM

Only for hot loops: distance tiles, simplex filtering, and bitset reduction.

Current source:

- `backends/asm/x86_64_l2_f32.S`
- `backends/asm/x86_64_dispatch.S`

Gate:

- CPUID feature detection;
- safe fallback;
- tiny unsafe/FFI surface;
- same barcode output as Rust and reference library.

## Triton

For GPU schedule construction and topology-guided kernels.

Current source: `backends/triton/topology_distance.py` contains an optional
Triton JIT pairwise-distance prototype. It is not imported by the core package.

Gate:

- dense fallback;
- schedule validation;
- dtype/device preservation;
- benchmark against dense SDPA or FlashAttention.

## PyTorch and TensorFlow

Adapters, not core dependencies. Use tensor exchange and model activation capture
without forcing heavy frameworks onto basic users.

Current benchmark: `benchmarks/benchmark_cuda_tensors.py` uses real CUDA tensors
through PyTorch, times GPU projection and distance work with CUDA events, and
then runs the current topology reduction on the selected activation cloud.
