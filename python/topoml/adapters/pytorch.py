from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..features import activation_signature
from ..topology import TopologySignature


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("PyTorch adapter requires the optional 'torch' dependency") from exc
    return torch


@dataclass(frozen=True)
class TorchTensorAdapter:
    """Convert PyTorch tensors to topology inputs while preserving metadata."""

    detach: bool = True

    def to_numpy(self, tensor: Any) -> np.ndarray:
        torch = _torch()
        if not torch.is_tensor(tensor):
            raise TypeError("tensor must be a torch.Tensor")
        value = tensor.detach() if self.detach else tensor
        return value.cpu().numpy()

    def from_numpy(self, values: np.ndarray, *, like: Any | None = None, dtype: Any | None = None, device: Any | None = None) -> Any:
        torch = _torch()
        target_dtype = dtype if dtype is not None else (like.dtype if like is not None else None)
        target_device = device if device is not None else (like.device if like is not None else None)
        tensor = torch.as_tensor(np.asarray(values), dtype=target_dtype)
        return tensor.to(target_device) if target_device is not None else tensor

    def activation_signature(
        self,
        activations: Any,
        *,
        radii: Sequence[float],
        max_dim: int = 1,
    ) -> TopologySignature:
        signature = activation_signature(self.to_numpy(activations), radii=radii, max_dim=max_dim)
        return TopologySignature(
            kind="torch_activation",
            values={
                "torch_rank": float(activations.ndim),
                "torch_device_cuda": float(getattr(activations.device, "type", "cpu") == "cuda"),
                **signature.values,
            },
        )


@dataclass(frozen=True)
class TorchActivationCapture:
    """Run a callable/module and summarize the tensor it returns."""

    model: Callable[..., Any]
    adapter: TorchTensorAdapter = TorchTensorAdapter()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def signature(self, *args: Any, radii: Sequence[float], max_dim: int = 1, **kwargs: Any) -> TopologySignature:
        torch = _torch()
        with torch.no_grad():
            output = self.model(*args, **kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return self.adapter.activation_signature(output, radii=radii, max_dim=max_dim)
