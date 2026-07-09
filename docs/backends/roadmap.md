# Backend Roadmap

The backend ladder is deliberate.

## Safe Rust

Default backend. Bounded exact PH through small dimensions. This is the truth
source for tests.

## C++

Portable native path for users who cannot use Rust extensions directly.

Current source: `backends/cpp/topoml_native.cpp` defines a C ABI for pairwise
distance, threshold-edge preprocessing, and H0 barcode construction.

Current verification:

- `python/topoml/native.py` compiles the C++ source into a shared library and
  loads it through `ctypes`.
- `python/tests/test_cpp_native_ctypes.py` compares C++ distances and threshold
  edges against NumPy and H0 barcode deaths against the Python reference.
- `benchmarks/benchmark_native_distance.py` writes a JSON timing/correctness
  smoke artifact for preprocessing and H0 barcode construction.
- GitHub Actions runs a dedicated `native C++ smoke` job on Ubuntu.

Gate:

- ABI documented;
- compile/load smoke against NumPy;
- H1/H2 barcode equivalence against Rust and external TDA baselines;
- sanitizer-clean test run.

## ASM

Only for hot loops: distance tiles, simplex filtering, and bitset reduction.

Current source:

- `backends/asm/x86_64_l2_f32.S`
- `backends/asm/x86_64_dispatch.S`

Current verification:

- `python/topoml/asm.py` compiles both assembly sources into a shared library on
  Linux x86-64 and loads them through `ctypes`.
- `python/tests/test_asm_native_ctypes.py` checks CPUID and XCR0 decoding,
  verifies that the AVX-512 symbol is exported, and compares dispatched
  L2-squared output against NumPy across multiple vector lengths.
- `benchmarks/benchmark_asm_distance.py` writes a JSON timing, dispatch-path,
  and CPU-feature smoke artifact.
- GitHub Actions runs this inside the native smoke job.

Gate:

- CPUID/XCR0 feature detection;
- safe scalar fallback when AVX-512 state is unavailable;
- tiny unsafe/FFI surface;
- same barcode output as Rust and reference library.

The active ASM claim is distance dispatch only. Barcode reduction, simplex
filtering, and broad PH acceleration remain gated until they have equivalence
tests and benchmark baselines.

## Triton

For GPU schedule construction and topology-guided kernels.

Current source: `backends/triton/topology_distance.py` contains an optional
Triton JIT pairwise-distance prototype. `python/topoml/triton.py` exposes a
lazy runtime wrapper that checks `torch`, `triton`, and CUDA availability before
launching the kernel.

Current runtime contract:

- `python/tests/test_triton_runtime.py` verifies that importing `topoml` does
  not import `torch` or `triton`.
- On CUDA machines with `torch` and `triton`, the same test compares
  `topoml.triton_pairwise_l2` against dense `torch.cdist` on a small fixture.
- On CPU-only machines, the test skips with the explicit missing runtime gate.

Current CPU semantic contract:

- `python/tests/test_gpu_backend_semantic_contract.py` fixes deterministic
  expected outputs for pairwise L2, threshold edges, row sums, and persistence
  image accumulation.
- The same test asserts the CUDA and Triton source files expose the expected
  public symbols.
- This is not a runtime GPU correctness gate. It defines the outputs that future
  CUDA extension runtime tests must match.

Current CUDA compile contract:

- `python/tests/test_cuda_nvcc_compile.py` compiles `backends/cuda/topology_distance.cu`
  and `backends/cuda/warp_reductions.cu` to object files when `nvcc` is present.
- The test skips on CPU-only machines without CUDA tooling, so normal users do
  not need a CUDA toolkit to run the package tests.
- On Windows it also requires the MSVC host compiler `cl.exe` on `PATH`, matching
  `nvcc`'s normal host-compiler requirement.
- The test is marked `cuda_compile`, so a CUDA-capable runner can enforce just
  this gate with `python -m pytest -m cuda_compile python/tests -q -rs`.
- This proves source-level CUDA compilability where CUDA tooling exists. It does
  not launch kernels or claim GPU runtime correctness.

Gate:

- topology-guided schedule validation;
- dtype/device preservation beyond the pairwise-L2 prototype;
- same-budget ablations;
- benchmark against dense SDPA or FlashAttention before any sparse-attention
  speedup claim.

## PyTorch and TensorFlow

Active optional adapters, not core dependencies. Use tensor exchange and model
activation capture without forcing heavy frameworks onto basic users.

Current API:

- `topoml.adapters.TorchTensorAdapter`
- `topoml.adapters.TorchActivationCapture`
- `topoml.adapters.TensorFlowTensorAdapter`
- `topoml.adapters.TensorFlowActivationCapture`

The adapters convert framework tensors to NumPy-backed topology inputs and
return topology signatures. They preserve dtype/device where the framework
allows it and keep import guards strict: importing `topoml` does not import
`torch` or `tensorflow`.

Current benchmark: `benchmarks/benchmark_cuda_tensors.py` uses real CUDA tensors
through PyTorch, times GPU projection and distance work with CUDA events, and
then runs the current topology reduction on the selected activation cloud.

Current CI integration:

- `python/tests/test_framework_adapters.py` runs against real CPU PyTorch and
  TensorFlow in the `ml integration / cpu frameworks` job.
- PyTorch coverage includes dtype/device round-trip, activation signatures, and
  a `torch.compile(..., backend="eager")` capture path.
- TensorFlow coverage includes eager tensor conversion and graph-mode
  `tf.function` parity for the same topology signature.
- `benchmarks/benchmark_ml_adapters.py` emits a JSON artifact with framework
  versions, dtype/device metadata, signature values, and timing smoke records.

These adapters are active for tensor conversion and activation topology
signatures. They do not claim framework-native persistent-homology kernels,
training-quality improvements, or sparse-attention speedups.
