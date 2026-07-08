from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .core import available_backends


BackendStatus = Literal["active", "planned"]


@dataclass(frozen=True)
class BackendSelectionResult:
    adapter: "BackendAdapter"
    available: bool
    message: str
    missing_gates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class BackendUnavailableError(RuntimeError):
    def __init__(self, adapter: "BackendAdapter"):
        gates = ", ".join(adapter.required_gates) or "no availability gate is registered"
        message = f"Backend {adapter.id!r} is unavailable; required gates: {gates}"
        super().__init__(message)
        self.adapter_id = adapter.id
        self.required_gates = adapter.required_gates
        self.warnings = adapter.warnings


@dataclass(frozen=True)
class BackendAdapter:
    id: str
    status: BackendStatus
    capabilities: tuple[str, ...]
    required_gates: tuple[str, ...]
    warnings: tuple[str, ...]
    _available: bool = field(repr=False)
    _executor: Callable[..., Any] | None = field(default=None, repr=False, compare=False)

    @property
    def available(self) -> bool:
        return self._available

    def availability(self) -> BackendSelectionResult:
        if self._available:
            return BackendSelectionResult(
                adapter=self,
                available=True,
                message=f"Backend {self.id!r} is available.",
                warnings=self.warnings,
            )
        return BackendSelectionResult(
            adapter=self,
            available=False,
            message=_unavailable_message(self),
            missing_gates=self.required_gates,
            warnings=self.warnings,
        )

    def validate_available(self) -> BackendSelectionResult:
        result = self.availability()
        if not result.available:
            raise BackendUnavailableError(self)
        return result

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self.validate_available()
        if self._executor is None:
            raise NotImplementedError(f"Backend {self.id!r} does not expose a generic execution entrypoint.")
        return self._executor(*args, **kwargs)


def backend_adapters() -> tuple[BackendAdapter, ...]:
    return tuple(_adapter_from_metadata(metadata) for metadata in available_backends())


def backend_adapter(backend_id: str) -> BackendAdapter:
    normalized = _normalize_backend_id(backend_id)
    for adapter in backend_adapters():
        if adapter.id == normalized:
            return adapter
    raise KeyError(f"Unknown backend adapter: {backend_id!r}")


def select_backend_adapter(backend_id: str, *, raise_unavailable: bool = True) -> BackendSelectionResult:
    adapter = backend_adapter(backend_id)
    result = adapter.availability()
    if raise_unavailable and not result.available:
        raise BackendUnavailableError(adapter)
    return result


def _adapter_from_metadata(metadata: Any) -> BackendAdapter:
    status: BackendStatus = "active" if metadata.active and metadata.available else "planned"
    return BackendAdapter(
        id=metadata.id,
        status=status,
        capabilities=metadata.capabilities,
        required_gates=metadata.gates,
        warnings=metadata.warnings,
        _available=metadata.active and metadata.available,
    )


def _normalize_backend_id(backend_id: str) -> str:
    return backend_id.strip().lower().replace("-", "_")


def _unavailable_message(adapter: BackendAdapter) -> str:
    gates = ", ".join(adapter.required_gates) or "no availability gate is registered"
    return f"Backend {adapter.id!r} is unavailable; required gates: {gates}"
