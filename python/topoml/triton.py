from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class TritonRuntimeStatus:
    torch_importable: bool
    triton_importable: bool
    cuda_available: bool
    available: bool
    message: str


class TritonBackendUnavailable(RuntimeError):
    pass


def triton_runtime_status() -> TritonRuntimeStatus:
    torch_spec = importlib.util.find_spec("torch")
    triton_spec = importlib.util.find_spec("triton")
    torch_importable = torch_spec is not None
    triton_importable = triton_spec is not None
    cuda_available = False

    if torch_importable:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_importable = False

    if triton_importable:
        try:
            import triton  # noqa: F401
        except Exception:
            triton_importable = False

    available = torch_importable and triton_importable and cuda_available
    if available:
        message = "Triton runtime is available with torch, triton, and CUDA."
    else:
        missing = []
        if not torch_importable:
            missing.append("torch import")
        if not triton_importable:
            missing.append("triton import")
        if not cuda_available:
            missing.append("CUDA device")
        message = "Triton runtime unavailable; missing " + ", ".join(missing) + "."

    return TritonRuntimeStatus(
        torch_importable=torch_importable,
        triton_importable=triton_importable,
        cuda_available=cuda_available,
        available=available,
        message=message,
    )


def triton_pairwise_l2(points: Any, *, block: int = 1024) -> Any:
    status = triton_runtime_status()
    if not status.available:
        raise TritonBackendUnavailable(status.message)
    module = _load_triton_distance_module()
    return module.pairwise_l2(points, block=block)


def _load_triton_distance_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    source = root / "backends" / "triton" / "topology_distance.py"
    spec = importlib.util.spec_from_file_location("topoml_triton_topology_distance", source)
    if spec is None or spec.loader is None:
        raise TritonBackendUnavailable(f"could not load Triton source at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
