# Backend Roadmap

The backend ladder is deliberate.

## Safe Rust

Default backend. Bounded exact PH through small dimensions. This is the truth
source for tests.

## C++

Portable native path for users who cannot use Rust extensions directly.

Gate:

- ABI documented;
- barcode equivalence against Rust;
- sanitizer-clean test run.

## ASM

Only for hot loops: distance tiles, simplex filtering, and bitset reduction.

Gate:

- CPUID feature detection;
- safe fallback;
- tiny unsafe/FFI surface;
- same barcode output as Rust and reference library.

## Triton

For GPU schedule construction and topology-guided kernels.

Gate:

- dense fallback;
- schedule validation;
- dtype/device preservation;
- benchmark against dense SDPA or FlashAttention.

## PyTorch and TensorFlow

Adapters, not core dependencies. Use tensor exchange and model activation capture
without forcing heavy frameworks onto basic users.

