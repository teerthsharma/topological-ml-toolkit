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

