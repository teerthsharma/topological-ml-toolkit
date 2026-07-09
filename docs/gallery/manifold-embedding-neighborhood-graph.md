# Manifold Embedding Neighborhood Graph

Embeddings are not only coordinates. They imply neighborhoods, cover cells,
quotient regions, drift pockets, and graph summaries that can be tested.

```mermaid
quadrantChart
  title Embedding Diagnostics
  x-axis Low topology signal --> High topology signal
  y-axis Low task actionability --> High task actionability
  quadrant-1 Prioritize
  quadrant-2 Explain
  quadrant-3 Archive
  quadrant-4 Investigate
  "Stable class cluster": [0.72, 0.86]
  "Rare bridge samples": [0.83, 0.55]
  "Random noise pocket": [0.28, 0.18]
  "Batch drift edge": [0.62, 0.74]
```

## Active APIs

Use `metric_cover`, `nerve_graph`, `mapper_graph`, `finite_orbit_signature`, and
`equivariance_residual` to turn embeddings into graph-ready diagnostics.

```python
cover = topoml.metric_cover(embedding, radius=0.5)
nerve = topoml.nerve_graph(cover)
mapper = topoml.mapper_graph(embedding, filter_values=loss, intervals=8, overlap=0.25)
```

```mermaid
graph TB
  E["Embedding cloud"] --> N["Neighborhood graph"]
  E --> C["Metric cover"]
  C --> R["Nerve graph"]
  E --> M["Mapper graph"]
  E --> Q["Quotient/orbit diagnostics"]
  R --> A["Routing and batching candidates"]
  M --> B["Model-debugging map"]
  Q --> D["Symmetry contract check"]
```

## Claim Boundary

The library gives executable graph diagnostics for embeddings. It does not claim
that every embedding is a manifold, that every graph is faithful, or that a
topology graph is better than UMAP, nearest-neighbor graphs, or label-based
inspection without benchmark evidence.
