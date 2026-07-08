# Topological ML Toolkit Design

## Goal

Build a polished Rust and Python library that makes persistent homology and
manifold embeddings useful to mainstream ML practitioners through tested APIs,
visual documentation, graph generation, benchmarks, and gated performance
backends.

## Audience

- data scientists who know sklearn/PyTorch but not algebraic topology;
- systems programmers who want bounded Rust and optional ASM/C++ acceleration;
- ML researchers testing topology-derived sparse schedules or model diagnostics.

## Architecture

The repo starts with safe Rust and Python. Optional C++, ASM, Triton, PyTorch,
and TensorFlow layers are adapters or backends, not required dependencies.

Core modules:

- `topoml-core`: no_std-friendly Rust persistent homology primitives.
- `python/topoml`: NumPy-first Python API and graph export.
- `docs`: MkDocs site with formulas, diagrams, and gallery.
- `scripts`: graph generation and documentation assets.
- `benchmarks`: reproducible benchmark runners and artifact contract.

## Documentation Standard

Every major concept must include:

- the formula;
- plain-language explanation;
- generated graph;
- practical ML use case;
- failure mode.

## Performance Standard

No speed claim is allowed without raw benchmark evidence. Sparse attention and
kernel claims must compare against dense optimized baselines plus same-budget
random and locality ablations.

## Git Standard

Avoid commit spam. Build and verify locally, then make one coherent initial
commit when the repo is shaped well enough to publish.

