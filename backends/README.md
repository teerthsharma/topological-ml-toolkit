# Backend Contracts

This directory contains backend contracts and early backend source files. The
files are real CUDA, Assembly, C++, and Triton source, but they are not selected
by the public package until correctness and benchmark gates pass.

## Source Inventory

- `cuda/topology_distance.cu`: CUDA pairwise distance and threshold-edge kernels.
- `cuda/warp_reductions.cu`: CUDA warp/block reductions and persistence-image accumulation.
- `asm/x86_64_l2_f32.S`: x86-64 scalar L2-squared ABI reference.
- `asm/x86_64_dispatch.S`: CPUID probes for AVX2 and AVX-512 gates.
- `cpp/topoml_native.cpp`: portable C++ C-ABI distance and threshold routines.
- `triton/topology_distance.py`: optional Triton JIT pairwise distance prototype.

## Current Native C++ Smoke Gate

The C++ backend now has a build-tested preprocessing gate:

- `python/topoml/native.py` compiles `cpp/topoml_native.cpp` into a shared library
  and loads it with `ctypes`.
- `python/topoml/asm.py` compiles the Linux x86-64 assembly probes into a shared
  library and loads them with `ctypes`.
- `python/tests/test_cpp_native_ctypes.py` verifies pairwise L2 distances and
  threshold edges against NumPy.
- `python/tests/test_asm_native_ctypes.py` verifies CPUID bit decoding and the
  scalar ASM L2-squared hot path against NumPy.
- `benchmarks/benchmark_native_distance.py` emits JSON timing artifacts.
- `benchmarks/benchmark_asm_distance.py` emits JSON timing and CPU feature
  artifacts.
- CI runs these on Ubuntu in the `native C++ smoke` job.

This is not yet a persistent-homology acceleration claim. It proves the portable
C ABI and assembly dispatch probes can be built and called correctly for
preprocessing.

## Backend Metadata

Every backend must report:

- backend id;
- dtype;
- device;
- CPU/GPU feature gates;
- build time;
- execute time;
- peak memory when measurable;
- warnings;
- reproducibility seed.

## ASM Contract

ASM backends are allowed only behind runtime feature detection and safe fallback.

Required checks:

- CPUID gate for AVX-512 or AVX2, currently smoke-tested through
  `topoml_cpuid_leaf7_ebx`, `topoml_has_avx2`, and `topoml_has_avx512f`;
- output barcode equivalence against safe Rust;
- no unchecked pointer arithmetic outside the FFI boundary;
- benchmark artifact before speed claim.

## Triton Contract

Triton kernels must follow compiler-kernel discipline:

- validate shape, dtype, device, and schedule layout before launch;
- keep dense fallback;
- report schedule-build time separately from kernel execution;
- compare against dense SDPA/FlashAttention when attention is involved.

## PyTorch and TensorFlow Contract

Framework adapters must not be required core dependencies. They convert tensors,
capture activations, and hand data to core topology routines.
