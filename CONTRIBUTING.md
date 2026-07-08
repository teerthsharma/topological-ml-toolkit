# Contributing

This project wants broad appeal without lowering the math or systems bar.

Before opening a PR:

1. Run Rust tests.
2. Run Python tests.
3. Generate docs graphs.
4. Add docs for new public behavior.
5. Add benchmark artifacts before adding speed claims.

Commands:

```powershell
cargo test -p topoml-core
$env:PYTHONPATH='python'; python -m pytest python/tests -q
$env:PYTHONPATH='python'; python scripts/generate_gallery.py
```

Keep commits coherent. One strong commit beats ten noisy commits.

