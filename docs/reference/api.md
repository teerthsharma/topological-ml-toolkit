# API Reference

## Rust

### `PointCloud<const D: usize>`

Validated point cloud with finite coordinates.

```rust
let cloud = PointCloud::<2>::new(vec![[0.0, 0.0], [1.0, 0.0]])?;
```

### `PersistenceConfig`

Bounded configuration for PH computation.

Fields:

- `max_homology_dim`: 0, 1, or 2.
- `max_points`: fail-fast cap before simplex explosion.
- `max_simplices`: fail-fast cap during complex construction.
- `max_radius`: filtration cutoff.
- `complex_kind`: `VietorisRips` or `Witness`.

### `persistent_homology`

Computes a persistence diagram from a point cloud.

```rust
let diagram = persistent_homology(&cloud, PersistenceConfig::builder().max_homology_dim(1).build())?;
let beta = diagram.betti_at(0.5);
```

### `time_delay_embedding`

Converts a scalar sequence into a point cloud.

```rust
let cloud = time_delay_embedding::<3>(&samples, 1)?;
```

## Python

### `topoml.persistent_homology(points, max_dim=1, max_radius=inf)`

Returns a `PersistenceDiagram`.

### `PersistenceDiagram.betti_at(radius)`

Returns `BettiNumbers(beta0, beta1, beta2)`.

### `PersistenceDiagram.to_plotly_trace(dimension=0)`

Returns a dictionary shaped for Plotly scatter charts.

### `topoml.time_delay_embedding(samples, dimension, tau=1)`

Returns a NumPy array of delay vectors.

### `topoml.PHFeaturizer(max_dim=1, radii=None, homology_dims=None)`

sklearn-style transformer that converts a batch of point clouds into a fixed
NumPy feature matrix.

```python
features = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.5, 1.0]).fit_transform(clouds)
```

The current active encoding is a Betti curve sampler:

\[
f = [\beta_0(r_1), \ldots, \beta_0(r_n), \beta_1(r_1), \ldots]
\]

### `topoml.BettiCurve(radii, homology_dims=(0, 1))`

Encodes existing `PersistenceDiagram` objects as fixed-width Betti curve samples.

```python
diagram = topoml.persistent_homology(points, max_dim=1, max_radius=2.0)
features = topoml.BettiCurve(radii=[0.0, 0.5, 1.0]).fit_transform([diagram])
```

### `topoml.PersistenceImage(width=16, height=16, sigma=0.1)`

Encodes finite persistence pairs into a flattened image-like vector using
birth-persistence coordinates and Gaussian kernels. Infinite bars are excluded
from the image because they do not have finite persistence.

```python
image = topoml.PersistenceImage(width=16, height=16).fit_transform([diagram])
```

### Topology signatures

`TopologySignature` is a small typed summary for point clouds, graphs, and model
activations.

```python
point_sig = topoml.point_cloud_signature(points, radii=[0.0, 0.5], max_dim=1)
graph_sig = topoml.graph_signature(adjacency)
activation_sig = topoml.activation_signature(activations, radii=[0.0, 1.0], max_dim=0)
```

Graph signatures report node count, undirected edge count, connected components,
and graph cycle rank:

\[
\beta_1(G) = |E| - |V| + c
\]

### TensorBundle interoperability

`TensorBundleSpec` describes a runtime tensor or vector space by basis labels,
diagonal metric values, tangent order, tangent-variable count, and dual status.

```python
xy = topoml.TensorBundleSpec(("x", "y"), (1.0, 1.0))
yz = topoml.TensorBundleSpec(("y", "z"), (1.0, -1.0), tangent_order=1)
ambient = topoml.interop_bundle(xy, yz)
```

The default interoperability rule is a union morphism:

\[
V_{xy} \cup V_{yz} = V_{xyz}
\]

`TensorAlgebraElement` stores dense coordinates tied to a bundle. `interop_add`
coerces mixed-space elements into the ambient bundle before addition:

```python
a = topoml.TensorAlgebraElement(xy, np.array([1.0, 2.0]))
b = topoml.TensorAlgebraElement(yz, np.array([3.0, 4.0]))
summed = topoml.interop_add(a, b)
```

`tensor_bundle_signature_features(bundle)` returns numeric metadata such as
rank, positive axes, negative axes, degenerate axes, tangent order, and dual
status.

### Topology-aware training

`TopologyAugmenter` appends persistent-homology features to ordinary ML feature
matrices:

```python
augmenter = topoml.TopologyAugmenter(radii=[0.0, 0.25, 0.5], max_dim=1)
features = augmenter.fit_transform(point_clouds, base_features=tabular_features)
```

`topological_sample_weights` returns mean-one weights from Betti-curve mass and
variation. The weights can be used as a pretraining or tabular-learning prior:

```python
weights = topoml.topological_sample_weights(point_clouds, radii=[0.0, 0.5, 1.0])
```

`TopologyRandomForestClassifier` is an executable dependency-light baseline. It
trains weighted random decision stumps on topology-augmented features:

```python
model = topoml.TopologyRandomForestClassifier(radii=[0.0, 0.5, 1.0], max_dim=1)
model.fit(point_clouds, labels, base_features=tabular_features)
predicted = model.predict(point_clouds, base_features=tabular_features)
```

### Backend adapters

`topoml.backend_adapters()` returns API-level backend contracts for active and
planned backends. Planned backends are discoverable but unavailable until their
correctness and benchmark gates pass.

```python
result = topoml.select_backend_adapter("triton", raise_unavailable=False)
print(result.available)
print(result.missing_gates)
```

Strict selection raises `BackendUnavailableError` for a planned backend:

```python
topoml.select_backend_adapter("asm_avx512")
```

### Framework tensor adapters

PyTorch and TensorFlow adapters are optional. Importing `topoml` does not import
`torch` or `tensorflow`; the adapters load those frameworks only when conversion
or capture methods run.

```python
from topoml.adapters import TorchTensorAdapter, TensorFlowTensorAdapter

torch_sig = TorchTensorAdapter().activation_signature(torch_activations, radii=[0.0, 1.0], max_dim=0)
tf_sig = TensorFlowTensorAdapter().activation_signature(tf_activations, radii=[0.0, 1.0], max_dim=0)
```

Capture helpers summarize callable/module outputs:

```python
from topoml.adapters import TorchActivationCapture

capture = TorchActivationCapture(model)
signature = capture.signature(batch, radii=[0.0, 1.0], max_dim=0)
```

These adapters are tensor-exchange and diagnostics APIs. They do not claim
accelerated persistent homology, sparse attention speedups, or training quality
improvements unless a benchmark artifact proves those claims.

### Topology family registry

`topology_families()` exposes the objective taxonomy as public data. The rendered
coverage dashboard lives at [Topology Family Coverage Matrix](../topology/family-coverage-matrix.md).
It separates docs-only families, prototype diagnostics, and benchmark evidence so
the library does not overclaim accelerated topology backends.

```python
for family in topoml.topology_families():
    print(family.id, family.status, family.api_surface)
```

`topology_family_coverage_matrix()` returns plain dictionaries suitable for
docs, dashboards, and CI checks.

### Prototype topology diagnostics

These APIs are active prototype diagnostics. They are implemented, tested, and
covered by the E2E claim gate. They do not claim accelerated backend behavior.

`topoml.metric_cover(points, radius)` builds one metric cover cell per point.
Each cell contains the indices within the radius.

`topoml.nerve_graph(cover)` returns a graph with one node per cover cell and an
edge when two cover cells share at least one point.

```python
cover = topoml.metric_cover(points, radius=0.25)
nerve = topoml.nerve_graph(cover)
```

`topoml.mapper_graph(points, filter_values, intervals, overlap, cluster_radius)`
builds a small Mapper-style graph from a scalar filter, overlapping intervals,
and threshold-connected components inside each pullback set.

```python
graph = topoml.mapper_graph(points, points[:, 0], intervals=4, overlap=0.25, cluster_radius=0.5)
```

`topoml.sheaf_consistency_residual(sections, restrictions)` evaluates local
agreement residuals across named sections and linear restriction maps.

```python
residual = topoml.sheaf_consistency_residual(
    {"layer0": x0, "layer1": x1},
    [("layer0", "layer1", restriction_matrix)],
)
```

### `topoml.write_dashboard(path, title, diagram=None, feature_matrix=None, metadata=None)`

Writes a self-contained HTML dashboard for local inspection or CI artifacts.
The exporter intentionally avoids heavy JavaScript dependencies, so it can be
opened from disk or uploaded as an artifact.

```python
topoml.write_dashboard(
    "artifacts/dashboard.html",
    title="Topology inspection",
    diagram=diagram,
    feature_matrix=features,
)
```

## Planned Public APIs

These are not shipped yet. They are design commitments with gates:

- `TritonScheduleBuilder`: topology-derived sparse schedule builder.
