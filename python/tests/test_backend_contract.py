from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys

import topoml


def test_safe_rust_python_cpp_and_framework_adapters_are_active_backends() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}

    for name in ["safe_rust", "python_reference", "cpp", "pytorch", "tensorflow"]:
        assert backends[name].active
        assert not backends[name].planned

    for name in ["safe_rust", "python_reference", "cpp"]:
        assert backends[name].available

    assert not backends["pytorch"].available
    assert not backends["tensorflow"].available
    assert topoml.select_backend("safe_rust").id == "safe_rust"
    assert topoml.select_backend("python_reference").id == "python_reference"
    assert topoml.select_backend("cpp").id == "cpp"
    assert topoml.select_backend("pytorch") is None
    assert topoml.select_backend("tensorflow") is None
    assert "persistent_homology_h0" in backends["cpp"].capabilities


def test_asm_avx512_is_active_optional_backend_with_cpu_gate() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}
    metadata = backends["asm_avx512"]

    assert metadata.active
    assert not metadata.planned
    assert metadata.available == (
        platform.system().lower() == "linux"
        and platform.machine().lower() in {"x86_64", "amd64"}
        and any(shutil.which(candidate) for candidate in ("cc", "gcc", "clang"))
    )
    assert "asm_l2_sq_f32" in metadata.capabilities
    assert "avx-512" in " ".join(metadata.gates).lower()
    assert "cpuid" in " ".join(metadata.warnings).lower()
    if metadata.available:
        assert topoml.select_backend("asm_avx512").id == "asm_avx512"
    else:
        assert topoml.select_backend("asm_avx512") is None


def test_triton_is_active_optional_backend_with_cuda_runtime_gate() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}
    metadata = backends["triton"]

    assert metadata.active
    assert not metadata.available
    assert not metadata.planned
    assert "triton_pairwise_l2" in metadata.capabilities
    assert "cuda" in " ".join(metadata.gates).lower()
    assert topoml.select_backend("triton") is None


def test_no_backends_are_planned_placeholders() -> None:
    backends = {backend.id: backend for backend in topoml.available_backends()}

    assert [name for name, metadata in backends.items() if metadata.planned] == []


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
