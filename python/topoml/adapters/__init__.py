"""Optional framework adapters.

Importing this package does not import PyTorch or TensorFlow. Each adapter
loads its framework lazily when tensor conversion or capture methods are used.
"""

from .pytorch import TorchActivationCapture, TorchTensorAdapter
from .tensorflow import TensorFlowActivationCapture, TensorFlowTensorAdapter

__all__ = [
    "TensorFlowActivationCapture",
    "TensorFlowTensorAdapter",
    "TorchActivationCapture",
    "TorchTensorAdapter",
]
