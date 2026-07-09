# Topological ML Toolkit

Topological ML Toolkit is a Rust and Python library for turning the shape of
data into features, diagnostics, and benchmarked ML pipeline components.

Topological machine learning asks a practical question: what structure remains
stable when data is noisy, embedded differently, sampled unevenly, or passed
through a model? Instead of looking only at coordinates, it measures connected
components, loops, voids, covers, neighborhoods, trajectories, and consistency
constraints. Those measurements can become normal ML features, model-debugging
signals, or routing rules.

The goal is not to make topology feel exotic. The goal is to make it usable by
people who already build sklearn, PyTorch, TensorFlow, and Rust systems.

## What It Can Do

- Convert point clouds into persistence diagrams and Betti numbers.
- Convert time series into delay-coordinate point clouds.
- Export graph-ready diagram traces for documentation and notebooks.
- Build fixed-width Betti-curve feature matrices with a sklearn-style
  `PHFeaturizer`.
- Keep active claims tied to tests and E2E benchmark artifacts.
- Expose backend contracts for Safe Rust, Python reference, C++, ASM AVX-512,
  Triton, PyTorch, and TensorFlow without pretending gated acceleration is done.

## Why Topology Helps ML

Topology gives ML engineers features that are stable under deformation. That is
useful when the exact coordinate system is less important than the structure:

- clusters merging or splitting;
- loops in sensor traces, embeddings, robotics paths, or periodic signals;
- holes and voids in sampled manifolds;
- shape changes across model layers;
- drift in activation geometry;
- covers and neighborhoods that can drive batching or routing;
- local consistency failures across distributed or multi-view systems.

A normal pipeline can use those signals as numeric features:

```text
raw data -> point cloud / embedding / trajectory -> topology -> feature matrix -> ML model
```

## Current Working Surface

Rust crate: `topoml-core`

- `PointCloud`
- `PersistenceConfig`
- `ComplexKind`
- `persistent_homology`
- `time_delay_embedding`

Python package: `topoml`

- `persistent_homology(points, max_dim, max_radius)`
- `time_delay_embedding(samples, dimension, tau)`
- `PersistenceDiagram.betti_at(radius)`
- `PersistenceDiagram.to_plotly_trace(dimension)`
- `PHFeaturizer(max_dim, radii, homology_dims)`
- `BettiCurve(radii, homology_dims)`
- `PersistenceImage(width, height, sigma)`
- `point_cloud_signature`, `graph_signature`, and `activation_signature`
- `TensorBundleSpec`, `TensorAlgebraElement`, `interop_bundle`, and
  `interop_add` for explicit tensor-space interoperability rules
- `TopologyAugmenter`, `topological_sample_weights`, and
  `TopologyRandomForestClassifier` for topology-augmented training baselines
- `make_sklearn_pipeline(estimator, ...)` for optional real sklearn `Pipeline`
  integration without importing sklearn during `import topoml`
- `metric_cover(points, radius)` and `nerve_graph(cover)`
- `mapper_graph(points, filter_values, intervals, overlap, cluster_radius)`
- `sheaf_consistency_residual(sections, restrictions)`
- `path_homotopy_signature`, `activation_strata`, `finite_orbit_signature`,
  `equivariance_residual`, `scott_fixed_point`, and
  `weak_convergence_residual`
- `finite_topology_signature`, `dynamical_signature`,
  `braid_crossing_signature`, and `mesh_euler_characteristic` for finite
  point-set, dynamics, braid, and low-dimensional diagnostics
- `write_dashboard(path, title, diagram, feature_matrix, metadata)`
- backend metadata and backend selection contracts
- strict backend adapters through `select_backend_adapter`

## Quickstart

```python
import numpy as np
import topoml

points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]])
diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)

print(diagram.betti_at(0.1).beta0)  # 3 connected components
print(diagram.betti_at(0.3).beta0)  # 2 connected components
print(diagram.betti_at(6.0).beta0)  # 1 connected component
```

Feature extraction for a normal ML pipeline:

```python
import numpy as np
import topoml

clouds = [
    np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]]),
    np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]),
]

features = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.15, 1.0]).fit_transform(clouds)
print(features)
```

Prototype topology diagnostics:

```python
cover = topoml.metric_cover(points, radius=0.25)
nerve = topoml.nerve_graph(cover)

mapper = topoml.mapper_graph(
    points,
    filter_values=points[:, 0],
    intervals=3,
    overlap=0.25,
    cluster_radius=0.5,
)

finite = topoml.finite_topology_signature({"a", "b"}, [set(), {"a"}, {"a", "b"}])
dynamics = topoml.dynamical_signature([3.0, 2.0, 1.2, 1.6, 1.1])
```

Static dashboard export:

```python
topoml.write_dashboard(
    "artifacts/dashboard.html",
    title="Topology inspection",
    diagram=diagram,
    feature_matrix=features,
    metadata={"dataset": "example"},
)
```

Topology-aware training baseline:

```python
model = topoml.TopologyRandomForestClassifier(
    radii=[0.0, 0.25, 0.5, 1.0],
    max_dim=1,
    random_state=7,
)
model.fit(point_clouds, labels, base_features=tabular_features)
print(model.score(point_clouds, labels, base_features=tabular_features))
```

Tensor-space interoperability:

```python
xy = topoml.TensorBundleSpec(("x", "y"), (1.0, 1.0))
yz = topoml.TensorBundleSpec(("y", "z"), (1.0, -1.0))
ambient = topoml.interop_bundle(xy, yz)
print(ambient.basis)  # ("x", "y", "z")
```

Optional sklearn pipeline:

```python
from sklearn.tree import DecisionTreeClassifier

pipeline = topoml.make_sklearn_pipeline(
    DecisionTreeClassifier(random_state=0),
    radii=[0.0, 0.15, 1.0],
    max_dim=0,
)
pipeline.fit(point_clouds, labels)
```

## Backend Roadmap

| Backend | Status | Purpose | Gate before claims |
| --- | --- | --- | --- |
| Safe Rust | Active | Bounded exact Vietoris-Rips PH | Known Betti fixtures pass |
| Python reference | Active | Data-science API and graphs | Python contract tests pass |
| C++ | Active H0 native path | Portable native extension path | H0 barcode equivalence vs Python and baseline fixtures |
| ASM AVX-512 | Active hardware-gated distance dispatch | L2-squared hot path | CPUID/XCR0 gate plus NumPy equivalence |
| CUDA | Active optional native runtime wrapper | Pairwise L2 and threshold-edge preprocessing prototype | `nvcc`, CUDA runtime/device, and NumPy equivalence |
| Triton | Active optional CUDA runtime wrapper | Pairwise L2 topology preprocessing prototype | `torch.cdist` parity when torch, Triton, and CUDA are available |
| PyTorch | Active optional adapter | Tensor/module adapters | CPU framework CI plus torch.compile-safe behavior |
| TensorFlow | Active optional adapter | Tensor adapters | CPU framework CI with eager and graph-mode parity |

Active hardware-gated means the implementation exists but only selects when the
machine and OS expose the required CPU state. Active optional adapter means the
implementation exists and is gated, but the heavy framework or CUDA dependency
is still optional. Runtime-gated backends must fail clearly rather than silently
falling back to a slower or different implementation.

```python
result = topoml.select_backend_adapter("triton", raise_unavailable=False)
print(result.available)       # False today
print(result.missing_gates)   # dense baseline, optional dependency, parity gates
```

## Benchmark Discipline

This repository treats performance statements as claims that must be executable.

The E2E claim benchmark currently verifies:

- known \(H_0\) cluster merges;
- known \(H_1\) square-cycle behavior;
- time-delay embedding output shape;
- fixed-width `PHFeaturizer` output;
- prototype finite topology, cover, nerve, Mapper, sheaf residual, homotopy,
  strata, orbit, equivariance, Scott fixed-point, weak convergence, sampled
  dynamics, braid-crossing, and mesh Euler diagnostics;
- optional real sklearn pipeline integration when scikit-learn is installed;
- graph-first gallery pages with explicit claim boundaries;
- self-contained HTML dashboard export;
- backend metadata separation between active, runtime-gated, and planned work;
- import safety for optional heavy stacks;
- real CPU PyTorch and TensorFlow adapter integration in CI;
- benchmark-smoke timing records for the active Python reference path.
- CI-gated parity against `ripser` and `GUDHI` on small Vietoris-Rips
  fixtures.

Manual CUDA tensor benchmark for GPU machines:

```powershell
python benchmarks/benchmark_cuda_tensors.py --out artifacts/cuda-tensor-topology.json
```

This benchmark allocates real CUDA tensors, runs a CUDA-backed PyTorch projection
and `torch.cdist`, times the GPU section with CUDA events, and then runs the
current topology reduction on the selected activation cloud. It refuses to emit
fake CUDA results when CUDA is unavailable.

Run it locally:

```powershell
python benchmarks/e2e_claims.py --json-out artifacts/e2e-claims.json --md-out artifacts/e2e-claims.md
python benchmarks/benchmark_tda_baselines.py --out artifacts/tda-baselines.json
```

## Local Checks

```powershell
cargo fmt --all -- --check
cargo clippy -p topoml-core --all-targets -- -D warnings
cargo test -p topoml-core
cargo check -p topoml-core --no-default-features
python -m pytest python/tests -q
python benchmarks/e2e_claims.py --json-out artifacts/e2e-claims.json --md-out artifacts/e2e-claims.md
python -m mkdocs build --strict
```

## Documentation

The GitHub Pages site teaches topology first, then ML usage, then backend and
benchmark contracts. It includes generated SVG diagrams for filtrations,
barcodes, Betti curves, and embeddings.

Docs are built with MkDocs:

```powershell
python scripts/generate_gallery.py
python -m mkdocs serve
```
