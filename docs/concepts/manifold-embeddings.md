# Manifold Embeddings

Manifold embedding turns raw observations into geometry that topology can read.

## Time-Delay Embedding

For a scalar time series \(x_t\), dimension \(D\), and delay \(\tau\):

\[
\Phi(t) = [x_t, x_{t+\tau}, \ldots, x_{t+(D-1)\tau}]
\]

This converts a waveform into a point cloud. Periodic signals often become loops.

![Time delay embedding](../assets/time_delay_embedding.svg)

## kNN Graph

For embedding points \(z_i\), define adjacency:

\[
A_{ij} = \mathbf{1}[d(z_i, z_j) \le \epsilon]
\]

This graph can feed Mapper, graph Laplacian features, or graph-based PH.

## Embedding Comparison

The docs will compare:

- PCA: linear, fast, weak for curled manifolds;
- Isomap: geodesic distances on a neighbor graph;
- LLE: local linear patches;
- t-SNE/UMAP: visualization-first, not metric-preserving proof.

The rule is simple: use graphs to explain, then use tests to prove.

## Tensor Bundles For ML Spaces

A model may expose many spaces at once: token embeddings, hidden activations,
attention-head projections, graph node features, tangent features from a
trajectory, and dual features from gradients. The toolkit represents those
spaces with `TensorBundleSpec`:

\[
V = (B, g, \mu, \nu)
\]

where \(B\) is a named basis, \(g\) is a diagonal metric signature, \(\mu\) is a
tangent order, and \(\nu\) is the number of tangent variables.

When two operations use different spaces, the active rule is explicit:

\[
\mathrm{interop}(a \in V, b \in W) =
\mathrm{op}(\iota_{V \to V \cup W}(a), \iota_{W \to V \cup W}(b))
\]

This is the Python runtime version of the DirectSum-style idea from tensor
algebra systems. It makes mixed embedding, activation, and gradient spaces
auditable before they become optimized Rust, C++, CUDA, Triton, or ASM kernels.

```python
hidden = topoml.TensorBundleSpec(("h1", "h2"), (1.0, 1.0))
grad = topoml.TensorBundleSpec(("h2", "loss"), (1.0, -1.0))
ambient = topoml.interop_bundle(hidden, grad)
```
