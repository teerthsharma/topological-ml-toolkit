from __future__ import annotations

import json
import subprocess
import sys

import pytest

import topoml
from topoml.backends import (
    BackendUnavailableError,
    backend_adapter,
    backend_adapters,
    select_backend_adapter,
)


PLANNED_BACKENDS: set[str] = set()
ACTIVE_HARDWARE_BACKENDS = {"asm_avx512"}
ACTIVE_OPTIONAL_BACKENDS = {"pytorch", "tensorflow"}
ACTIVE_GPU_BACKENDS = {"cuda", "triton"}


def test_import_topoml_does_not_import_optional_backend_stacks() -> None:
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


def test_no_backend_adapters_are_planned_placeholders() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    assert PLANNED_BACKENDS == set()
    assert [backend_id for backend_id, adapter in adapters.items() if adapter.status == "planned"] == []


def test_active_backend_adapters_are_metadata_available() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    assert adapters["safe_rust"].status == "active"
    assert adapters["python_reference"].status == "active"
    assert adapters["cpp"].status == "active"
    assert adapters["asm_avx512"].status == "active"
    assert adapters["cuda"].status == "active"
    assert adapters["triton"].status == "active"
    assert adapters["pytorch"].status == "active"
    assert adapters["tensorflow"].status == "active"
    assert adapters["safe_rust"].validate_available().available
    assert adapters["python_reference"].validate_available().available
    assert adapters["cpp"].validate_available().available


def test_active_optional_framework_adapters_report_dependency_availability() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    for backend_id in ACTIVE_OPTIONAL_BACKENDS:
        adapter = adapters[backend_id]

        assert adapter.status == "active"
        assert "optional" in " ".join(adapter.required_gates).lower()
        if adapter.available:
            assert adapter.availability().available
        else:
            result = select_backend_adapter(backend_id, raise_unavailable=False)
            assert result.adapter.status == "active"
            assert not result.available
            assert "required gates" in result.message.lower()


def test_active_hardware_adapters_report_runtime_gate_availability() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    for backend_id in ACTIVE_HARDWARE_BACKENDS:
        adapter = adapters[backend_id]

        assert adapter.status == "active"
        assert "cpuid" in " ".join(adapter.required_gates + adapter.warnings).lower()
        if adapter.available:
            assert adapter.availability().available
        else:
            result = select_backend_adapter(backend_id, raise_unavailable=False)
            assert result.adapter.status == "active"
            assert not result.available
            assert "required gates" in result.message.lower()


def test_active_gpu_adapters_report_runtime_gate_availability() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    for backend_id in ACTIVE_GPU_BACKENDS:
        adapter = adapters[backend_id]

        assert adapter.status == "active"
        assert "cuda" in " ".join(adapter.required_gates).lower()
        if adapter.available:
            assert adapter.availability().available
        else:
            result = select_backend_adapter(backend_id, raise_unavailable=False)
            assert result.adapter.status == "active"
            assert not result.available
            assert "required gates" in result.message.lower()


def test_selecting_triton_backend_reports_runtime_gate_result() -> None:
    result = select_backend_adapter("triton", raise_unavailable=False)

    assert result.adapter.id == "triton"
    if result.available:
        assert "available" in result.message.lower()
    else:
        assert "triton" in result.message.lower()
        assert "required gates" in result.message.lower()


def test_selecting_unavailable_triton_backend_can_raise_clear_error() -> None:
    if backend_adapter("triton").available:
        assert select_backend_adapter("triton").available
    else:
        with pytest.raises(BackendUnavailableError, match="triton.*required gates"):
            select_backend_adapter("triton")


def test_backend_adapters_are_exported_from_topoml() -> None:
    assert topoml.backend_adapter("cpp").id == "cpp"
    assert topoml.select_backend_adapter("tensorflow", raise_unavailable=False).adapter.id == "tensorflow"
