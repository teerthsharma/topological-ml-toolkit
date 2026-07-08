# Benchmark Policy

Benchmarks are generated from scripts. Tables in docs are summaries, not hand
typed truth.

## Workloads

- known clusters with known H0;
- circles with known H1;
- spheres and tori when H2 is enabled;
- Swiss roll embeddings;
- model activation point clouds;
- Q/K/V captures for sparse attention.

## Output Contract

Each benchmark artifact records:

- backend id;
- dtype;
- CPU/GPU/device details;
- seed;
- build time;
- execution time;
- peak memory when measurable;
- warning list;
- correctness metric.

