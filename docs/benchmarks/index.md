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

Full benchmark artifacts should record:

- backend id;
- dtype;
- CPU/GPU/device details;
- seed;
- build time;
- execution time;
- peak memory when measurable;
- warning list;
- correctness metric.

The current E2E claim artifact is smaller because it is a correctness and API
claim gate, not a full performance study. It records deterministic evidence for
active public behavior and marks timing rows as smoke records rather than
speedup claims.
