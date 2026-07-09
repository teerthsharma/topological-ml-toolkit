# Benchmarks

Benchmarks must generate raw artifacts before docs summarize numbers.

Planned commands:

```powershell
$env:PYTHONPATH='python'; python benchmarks/benchmark_python.py --out artifacts/python-ph.json
python benchmarks/benchmark_native_distance.py --out artifacts/native-distance.json --points 8 16
python benchmarks/benchmark_asm_distance.py --out artifacts/asm-distance.json --points 8 16 --dims 8
python benchmarks/benchmark_ml_adapters.py --out artifacts/ml-adapters.json
python benchmarks/benchmark_triton_schedule.py --out artifacts/triton-schedule.json
python benchmarks/benchmark_tda_baselines.py --out artifacts/tda-baselines.json
cargo bench -p topoml-core
```

Baselines:

- safe Rust core;
- Python wrapper;
- CI-gated `ripser`;
- CI-gated `GUDHI`;
- simple Union-Find H0 baseline;
- dense SDPA/FlashAttention for sparse attention work.

The current `ripser` and `GUDHI` gate is intentionally narrow: it proves that
small deterministic Vietoris-Rips fixtures use the same filtration convention
and produce the same H0 deaths and H1 Betti counts. It is not a speed claim.
