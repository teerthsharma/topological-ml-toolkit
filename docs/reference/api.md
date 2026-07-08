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

## Planned Public APIs

These are not shipped yet. They are design commitments with gates:

- `PHFeaturizer`: sklearn-compatible transformer.
- `BettiCurve`: fixed vector curve sampler.
- `PersistenceImage`: image feature encoder.
- `TopologySignature`: backend-independent feature summary.
- `TorchActivationCapture`: PyTorch adapter.
- `TensorFlowActivationCapture`: TensorFlow adapter.
- `TritonScheduleBuilder`: topology-derived sparse schedule builder.

