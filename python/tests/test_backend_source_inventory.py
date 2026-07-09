from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_planned_backend_source_files_exist():
    expected = [
        ROOT / "backends" / "cuda" / "topology_distance.cu",
        ROOT / "backends" / "cuda" / "warp_reductions.cu",
        ROOT / "backends" / "asm" / "x86_64_l2_f32.S",
        ROOT / "backends" / "asm" / "x86_64_dispatch.S",
        ROOT / "backends" / "cpp" / "topoml_native.cpp",
        ROOT / "backends" / "triton" / "topology_distance.py",
        ROOT / "python" / "topoml" / "asm.py",
        ROOT / "python" / "tests" / "test_asm_native_ctypes.py",
        ROOT / "benchmarks" / "benchmark_asm_distance.py",
    ]

    for path in expected:
        assert path.exists(), path
        assert path.stat().st_size > 0, path


def test_optional_backend_sources_are_not_package_imports():
    import topoml

    exported = set(topoml.__all__)

    assert "pairwise_l2" not in exported
    assert "topoml_pairwise_l2_f64" not in exported
    assert "topoml_l2_sq_f32_asm" not in exported
