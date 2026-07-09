# Backend Roadmap

The backend ladder is deliberate.

## Safe Rust

Default backend. Bounded exact PH through small dimensions. This is the truth
source for tests.

## C++

Portable native path for users who cannot use Rust extensions directly.

Current source: `backends/cpp/topoml_native.cpp` defines a C ABI for pairwise
distance and threshold-edge preprocessing.

Current verification:

- `python/topoml/native.py` compiles the C++ source into a shared library and
  loads it through `ctypes`.
- `python/tests/test_cpp_native_ctypes.py` compares C++ distances and threshold
  edges against NumPy.
- `benchmarks/benchmark_native_distance.py` writes a JSON timing/correctness
  smoke artifact.
- GitHub Actions runs a dedicated `native C++ smoke` job on Ubuntu.

Gate:

- ABI documented;
- compile/load smoke against NumPy;
- barcode equivalence against Rust;
- sanitizer-clean test run.

## ASM

Only for hot loops: distance tiles, simplex filtering, and bitset reduction.

Current source:

- `backends/asm/x86_64_l2_f32.S`
- `backends/asm/x86_64_dispatch.S`

Current verification:

- `python/topoml/asm.py` compiles both assembly sources into a shared library on
  Linux x86-64 and loads them through `ctypes`.
- `python/tests/test_asm_native_ctypes.py` checks CPUID bit decoding and compares
  the scalar `topoml_l2_sq_f32_asm` routine against NumPy.
- `benchmarks/benchmark_asm_distance.py` writes a JSON timing and CPU-feature
  smoke artifact.
- GitHub Actions runs this inside the native smoke job.

Gate:

- CPUID feature detection;
- safe fallback;
- tiny unsafe/FFI surface;
- same barcode output as Rust and reference library.

## Triton

For GPU schedule construction and topology-guided kernels.

Current source: `backends/triton/topology_distance.py` contains an optional
Triton JIT pairwise-distance prototype. It is not imported by the core package.

Current CPU semantic contract:

- `python/tests/test_gpu_backend_semantic_contract.py` fixes deterministic
  expected outputs for pairwise L2, threshold edges, row sums, and persistence
  image accumulation.
- The same test asserts the CUDA and Triton source files expose the expected
  public symbols.
- This is not a runtime GPU correctness gate. It defines the outputs that future
  CUDA/Triton runtime tests must match.

Gate:

- dense fallback;
- schedule validation;
- dtype/device preservation;
- benchmark against dense SDPA or FlashAttention.

## PyTorch and TensorFlow

Adapters, not core dependencies. Use tensor exchange and model activation capture
without forcing heavy frameworks onto basic users.

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
