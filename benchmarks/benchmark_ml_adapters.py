from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def _record(name: str, fn) -> dict:
    start = time.perf_counter()
    evidence = fn()
    evidence["seconds"] = round(time.perf_counter() - start, 9)
    return {"name": name, "evidence": evidence}


def _torch_record() -> dict:
    import torch
    from topoml.adapters import TorchActivationCapture, TorchTensorAdapter

    tensor = torch.tensor([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=torch.float32)
    adapter = TorchTensorAdapter()
    eager = TorchActivationCapture(lambda x: x + 1.0).signature(tensor, radii=[0.0, 1.1], max_dim=0)
    compiled_values = None
    if hasattr(torch, "compile"):
        compiled = torch.compile(torch.nn.Identity(), backend="eager")
        compiled_values = TorchActivationCapture(compiled).signature(tensor, radii=[0.0, 1.1], max_dim=0).values
    restored = adapter.from_numpy(adapter.to_numpy(tensor), like=tensor)
    return {
        "torch_version": torch.__version__,
        "dtype": str(restored.dtype),
        "device": str(restored.device),
        "eager_values": eager.values,
        "compile_values": compiled_values,
    }


def _tensorflow_record() -> dict:
    import tensorflow as tf
    from topoml.adapters import TensorFlowActivationCapture, TensorFlowTensorAdapter

    tensor = tf.constant([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=tf.float32)
    adapter = TensorFlowTensorAdapter()
    eager = adapter.activation_signature(tensor, radii=[0.0, 1.1], max_dim=0)

    @tf.function
    def graph_model(x):
        return x + tf.constant(1.0, dtype=x.dtype)

    graph = TensorFlowActivationCapture(graph_model).signature(tensor, radii=[0.0, 1.1], max_dim=0)
    restored = adapter.from_numpy(adapter.to_numpy(tensor), dtype=tensor.dtype)
    return {
        "tensorflow_version": tf.__version__,
        "dtype": restored.dtype.name,
        "shape": list(restored.shape),
        "eager_values": eager.values,
        "graph_values": graph.values,
        "graph_matches_eager": graph.values == eager.values,
    }


def run(out: Path) -> dict:
    rows = [
        _record("torch adapter eager and compile-safe capture", _torch_record),
        _record("tensorflow adapter eager and graph capture", _tensorflow_record),
    ]
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rows": rows,
        "claim_scope": "real CPU framework adapter integration; not accelerated PH or model-quality evidence",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/ml-adapters.json"))
    args = parser.parse_args()
    run(args.out)


if __name__ == "__main__":
    main()
