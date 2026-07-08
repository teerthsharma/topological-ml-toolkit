from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from ..features import activation_signature
from ..topology import TopologySignature


def _tf() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise RuntimeError("TensorFlow adapter requires the optional 'tensorflow' dependency") from exc
    return tf


@dataclass(frozen=True)
class TensorFlowTensorAdapter:
    """Convert TensorFlow tensors to topology inputs with lazy imports."""

    def to_numpy(self, tensor: Any) -> np.ndarray:
        tf = _tf()
        if not tf.is_tensor(tensor):
            raise TypeError("tensor must be a TensorFlow Tensor")
        return tensor.numpy()

    def from_numpy(self, values: np.ndarray, *, dtype: Any | None = None) -> Any:
        tf = _tf()
        return tf.convert_to_tensor(np.asarray(values), dtype=dtype)

    def activation_signature(
        self,
        activations: Any,
        *,
        radii: Sequence[float],
        max_dim: int = 1,
    ) -> TopologySignature:
        signature = activation_signature(self.to_numpy(activations), radii=radii, max_dim=max_dim)
        return TopologySignature(
            kind="tensorflow_activation",
            values={"tensorflow_rank": float(len(activations.shape)), **signature.values},
        )


@dataclass(frozen=True)
class TensorFlowActivationCapture:
    """Run a callable/layer and summarize the TensorFlow tensor it returns."""

    model: Callable[..., Any]
    adapter: TensorFlowTensorAdapter = TensorFlowTensorAdapter()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def signature(self, *args: Any, radii: Sequence[float], max_dim: int = 1, **kwargs: Any) -> TopologySignature:
        output = self.model(*args, **kwargs)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return self.adapter.activation_signature(output, radii=radii, max_dim=max_dim)
