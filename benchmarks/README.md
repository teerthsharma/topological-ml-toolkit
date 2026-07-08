# Benchmarks

Benchmarks must generate raw artifacts before docs summarize numbers.

Planned commands:

```powershell
$env:PYTHONPATH='python'; python benchmarks/benchmark_python.py --out artifacts/python-ph.json
cargo bench -p topoml-core
```

Baselines:

- safe Rust core;
- Python wrapper;
- optional `ripser`;
- optional `GUDHI`;
- simple Union-Find H0 baseline;
- dense SDPA/FlashAttention for sparse attention work.

