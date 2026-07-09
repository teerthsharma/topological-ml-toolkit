# Backend Runtime Gates

Accelerated topology must be honest about what ran. The toolkit separates active
source code, optional runtime gate availability, and benchmark claims.

```mermaid
stateDiagram-v2
  [*] --> ImportTopoml
  ImportTopoml --> SafeRust: always available
  ImportTopoml --> PythonReference: always available
  ImportTopoml --> NativeProbe: user asks for C++/ASM/CUDA/Triton
  NativeProbe --> CppActive: compiler and shared library load
  NativeProbe --> AsmActive: Linux x86-64 plus CPUID/XCR0 gate
  NativeProbe --> CudaActive: nvcc, host compiler, runtime, device
  NativeProbe --> TritonActive: torch, triton, CUDA device
  NativeProbe --> Fallback: missing gate
  CppActive --> Evidence
  AsmActive --> Evidence
  CudaActive --> Evidence
  TritonActive --> Evidence
  Fallback --> Evidence
  Evidence --> [*]
```

## Active APIs

- `topoml.backend_adapters()`
- `topoml.select_backend_adapter(name, raise_unavailable=False)`
- `topoml.build_cpp_native_backend(path)`
- `topoml.build_asm_native_backend(path)`
- `topoml.build_cuda_native_backend(path)`
- `topoml.triton_runtime_status()`

```mermaid
sequenceDiagram
  participant User
  participant Adapter as Backend adapter
  participant Gate as Runtime gate
  participant Kernel as Native/GPU kernel
  participant Ref as Safe fallback
  User->>Adapter: request backend
  Adapter->>Gate: check dependencies and hardware
  Gate-->>Adapter: available or missing gates
  alt available
    Adapter->>Kernel: execute scoped kernel
  else unavailable
    Adapter->>Ref: return fallback path or explicit error
  end
```

## Claim Boundary

The active accelerated implementations are scoped: C++ preprocessing/H0, ASM
distance dispatch, CUDA pairwise L2 and threshold edges, Triton pairwise L2, and
framework tensor adapters. Broad GPU persistent homology, sparse attention
speedups, and training-quality improvements remain gated by future benchmarks.
