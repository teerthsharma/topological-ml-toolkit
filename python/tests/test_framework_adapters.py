import json
import importlib
import subprocess
import sys

import numpy as np
import pytest


def test_framework_adapter_modules_do_not_import_heavy_stacks_on_topoml_import():
    code = """
import json
import sys
import topoml
print(json.dumps({name: name in sys.modules for name in ["torch", "tensorflow"]}))
"""
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert json.loads(result.stdout) == {"torch": False, "tensorflow": False}


def test_torch_tensor_adapter_preserves_dtype_device_and_signature_when_available():
    torch = pytest.importorskip("torch")
    from topoml.adapters import TorchActivationCapture, TorchTensorAdapter

    adapter = TorchTensorAdapter()
    tensor = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)

    restored = adapter.from_numpy(adapter.to_numpy(tensor), like=tensor)
    signature = adapter.activation_signature(tensor, radii=[0.0, 1.1], max_dim=0)
    capture = TorchActivationCapture(lambda x: x + 1.0)
    captured = capture.signature(tensor, radii=[0.0, 1.1], max_dim=0)

    assert restored.dtype == tensor.dtype
    assert restored.device == tensor.device
    assert signature.kind == "torch_activation"
    assert signature.values["samples"] == 4.0
    assert signature.values["beta0@1.1"] == 1.0
    assert captured.kind == "torch_activation"


def test_torch_activation_capture_accepts_compile_safe_callable_when_available():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is unavailable")
    from topoml.adapters import TorchActivationCapture

    class Shift(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    tensor = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    compiled = torch.compile(Shift(), backend="eager")
    signature = TorchActivationCapture(compiled).signature(tensor, radii=[0.0, 1.1], max_dim=0)

    assert signature.kind == "torch_activation"
    assert signature.values["torch_rank"] == 3.0
    assert signature.values["beta0@1.1"] == 1.0


def test_tensorflow_tensor_adapter_eager_and_function_parity_when_available():
    try:
        tf = importlib.import_module("tensorflow")
    except Exception as exc:
        pytest.skip(f"tensorflow is not importable in this environment: {exc}")
    from topoml.adapters import TensorFlowActivationCapture, TensorFlowTensorAdapter

    adapter = TensorFlowTensorAdapter()
    tensor = tf.constant([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=tf.float32)

    restored = adapter.from_numpy(adapter.to_numpy(tensor), dtype=tensor.dtype)
    eager_signature = adapter.activation_signature(tensor, radii=[0.0, 1.1], max_dim=0)

    @tf.function
    def model(x):
        return x + tf.constant(1.0, dtype=x.dtype)

    captured = TensorFlowActivationCapture(model).signature(tensor, radii=[0.0, 1.1], max_dim=0)

    assert restored.dtype == tensor.dtype
    assert np.asarray(restored.numpy()).shape == tuple(tensor.shape)
    assert eager_signature.kind == "tensorflow_activation"
    assert eager_signature.values["beta0@1.1"] == 1.0
    assert captured.kind == "tensorflow_activation"
    assert captured.values == eager_signature.values
