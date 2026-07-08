# Topological ML Toolkit

Graph-first persistent homology and manifold embedding tools for ML practitioners.

This repository is built to turn topology into normal ML engineering:

1. Load data as point clouds, time series, embeddings, images, or model activations.
2. Compute persistent homology and manifold summaries with bounded, tested code.
3. Export diagrams, barcodes, Betti curves, and feature vectors.
4. Feed those features into familiar Python, PyTorch, TensorFlow, or sklearn workflows.
5. Benchmark every speed claim against declared baselines before saying it is fast.

## Why This Exists

Many topology repos look like research notes: clever math, thin docs, no graphs, no
clear baselines, and no path for normal data scientists. This project takes the
opposite route. The math is explained with pictures and derivations. The code has
small public APIs. Speedups are treated as claims that need evidence.

## Current Working Surface

- Rust crate: `topoml-core`
  - `PointCloud`
  - `PersistenceConfig`
  - `ComplexKind`
  - `persistent_homology`
  - `time_delay_embedding`
- Python package: `topoml`
  - `persistent_homology(points, max_dim, max_radius)`
  - `time_delay_embedding(samples, dimension, tau)`
  - graph-ready `PersistenceDiagram.to_plotly_trace()`
- Docs site: `mkdocs-material` with generated SVG graphs
- CI: Rust tests, Python tests, and docs graph generation

## Engineering Bar

This repo follows the upstream-style standard from serious PyTorch/Triton work:

- dense or safe fallback first;
- compression/sparsity only after calibration;
- strict shape, dtype, device, and budget validation;
- benchmark runner checked into the repo;
- no hand-maintained performance table without raw artifacts;
- optional dependencies stay optional;
- docs explain both the win path and the failure path.

## Backend Roadmap

| Backend | Status | Purpose | Gate before claims |
| --- | --- | --- | --- |
| Safe Rust | Active | Bounded exact Vietoris-Rips PH | Known Betti fixtures pass |
| Python | Active | Data-science API and graphs | Python contract tests pass |
| C++ | Planned | Portable native extension path | Barcode equivalence vs Rust and ripser/GUDHI |
| ASM AVX-512 | Planned | Distance and reduction hot paths | CPUID gate plus correctness equivalence |
| Triton | Planned | GPU schedule kernels for topology-guided ML | Dense SDPA/FlashAttention baseline and same-budget ablations |
| PyTorch | Planned | Tensor/module adapters | Dense fallback and torch.compile-safe behavior |
| TensorFlow | Planned | Tensor adapters | Eager and graph-mode parity |

## Quickstart

```python
import numpy as np
import topoml

points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]])
diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)

print(diagram.betti_at(0.1).beta0)  # 3 components
print(diagram.betti_at(0.3).beta0)  # 2 components
print(diagram.to_plotly_trace())
```

## Local Checks

```powershell
cargo test -p topoml-core
$env:PYTHONPATH='python'; python -m pytest python/tests -q
python scripts/generate_gallery.py
```

