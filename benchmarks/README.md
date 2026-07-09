# Benchmarks

Benchmarks must generate raw artifacts before docs summarize numbers.

Planned commands:

```powershell
$env:PYTHONPATH='python'; python benchmarks/benchmark_python.py --out artifacts/python-ph.json
python benchmarks/benchmark_native_distance.py --out artifacts/native-distance.json --points 8 16
python benchmarks/benchmark_asm_distance.py --out artifacts/asm-distance.json --points 8 16 --dims 8
python benchmarks/benchmark_ml_adapters.py --out artifacts/ml-adapters.json
cargo bench -p topoml-core
```

Baselines:

- safe Rust core;
- Python wrapper;
- optional `ripser`;
- optional `GUDHI`;
- simple Union-Find H0 baseline;
- dense SDPA/FlashAttention for sparse attention work.
