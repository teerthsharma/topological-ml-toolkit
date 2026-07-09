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


PLANNED_BACKENDS = {"asm_avx512", "triton", "pytorch", "tensorflow"}


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


def test_planned_backend_adapters_are_discoverable_with_contract_fields() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    assert PLANNED_BACKENDS.issubset(adapters)
    for backend_id in PLANNED_BACKENDS:
        adapter = adapters[backend_id]

        assert adapter.status == "planned"
        assert adapter.capabilities
        assert adapter.required_gates
        assert adapter.warnings


def test_active_backend_adapters_are_metadata_available() -> None:
    adapters = {adapter.id: adapter for adapter in backend_adapters()}

    assert adapters["safe_rust"].status == "active"
    assert adapters["python_reference"].status == "active"
    assert adapters["cpp"].status == "active"
    assert adapters["safe_rust"].validate_available().available
    assert adapters["python_reference"].validate_available().available
    assert adapters["cpp"].validate_available().available


def test_selecting_unavailable_planned_backend_returns_clear_result() -> None:
    result = select_backend_adapter("pytorch", raise_unavailable=False)

    assert not result.available
    assert result.adapter.id == "pytorch"
    assert "pytorch" in result.message.lower()
    assert "required gates" in result.message.lower()


def test_selecting_unavailable_planned_backend_can_raise_clear_error() -> None:
    with pytest.raises(BackendUnavailableError, match="asm_avx512.*required gates"):
        select_backend_adapter("asm-avx512")


def test_planned_backend_adapters_cannot_execute_without_gates() -> None:
    for backend_id in PLANNED_BACKENDS:
        adapter = backend_adapter(backend_id)

        with pytest.raises(BackendUnavailableError, match=backend_id):
            adapter.execute()


def test_backend_adapters_are_exported_from_topoml() -> None:
    assert topoml.backend_adapter("cpp").id == "cpp"
    assert topoml.select_backend_adapter("tensorflow", raise_unavailable=False).adapter.id == "tensorflow"
