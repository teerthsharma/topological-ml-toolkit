# Persistent Homology Feature Vectors

ML models need fixed-width vectors. Barcodes have variable length. This page
shows the bridge.

## Total Persistence

For barcode lifetimes \(p_i = d_i - b_i\), total persistence of order \(q\) is:

\[
TP_q = \sum_i p_i^q
\]

Use \(q=1\) for total lifetime and \(q=2\) to emphasize long-lived features.

## Persistence Entropy

Normalize lifetimes:

\[
\hat p_i = \frac{p_i}{\sum_j p_j}
\]

Then:

\[
H = -\sum_i \hat p_i \log(\hat p_i)
\]

Low entropy means one or two dominant topological structures. High entropy
means many similar structures.

## Betti Curve Samples

Pick radii \(r_1, \ldots, r_N\). Use:

\[
f = [\beta_0(r_1), \ldots, \beta_0(r_N), \beta_1(r_1), \ldots, \beta_1(r_N)]
\]

This is simple, interpretable, and graphable.

## Chebyshev Radii

Uniform samples waste points near flat regions. Chebyshev nodes concentrate near
the ends:

\[
t_j = \frac{1}{2}\left(1 - \cos\frac{(2j - 1)\pi}{2N}\right)
\]

Map \(t_j\) into your radius range and sample the Betti curve there.

