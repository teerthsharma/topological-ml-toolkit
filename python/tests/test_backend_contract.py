from __future__ import annotations

import json
import subprocess
import sys

import topoml


def test_safe_rust_and_python_reference_are_active_backends() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}

    for name in ["safe_rust", "python_reference", "cpp"]:
        assert backends[name].active
        assert backends[name].available
        assert not backends[name].planned

    assert topoml.select_backend("safe_rust").id == "safe_rust"
    assert topoml.select_backend("python_reference").id == "python_reference"
    assert topoml.select_backend("cpp").id == "cpp"
    assert "persistent_homology_h0" in backends["cpp"].capabilities


def test_planned_backends_expose_gates_and_are_not_selectable() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}

    for name in ["asm_avx512", "triton", "pytorch", "tensorflow"]:
        metadata = backends[name]

        assert not metadata.active
        assert not metadata.available
        assert metadata.planned
        assert metadata.gates
        assert topoml.select_backend(name) is None

    assert "cpuid" in " ".join(backends["asm_avx512"].gates).lower()
    assert "correctness" in " ".join(backends["asm_avx512"].warnings).lower()


def test_import_topoml_does_not_import_heavy_frameworks() -> None:
    code = """
import json
import sys
import topoml
print(json.dumps({name: name in sys.modules for name in ["torch", "tensorflow", "triton"]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {
        "torch": False,
        "tensorflow": False,
        "triton": False,
    }
