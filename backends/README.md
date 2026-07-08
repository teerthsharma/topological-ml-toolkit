# Backend Contracts

This directory documents backend contracts before low-level code lands.

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

- CPUID gate for AVX-512 or AVX2;
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

