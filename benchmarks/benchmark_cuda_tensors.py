from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import topoml


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional local environment
        raise SystemExit("torch is required for the CUDA tensor benchmark") from exc
    return torch


def run_cuda_tensor_case(
    *,
    batch: int,
    tokens: int,
    hidden: int,
    projected: int,
    topology_points: int,
    max_dim: int,
    max_radius: float,
    seed: int,
) -> dict:
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to produce fake CUDA benchmark results")

    device = torch.device("cuda")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    layer = torch.nn.Linear(hidden, projected, bias=False, device=device)
    x = torch.randn(batch, tokens, hidden, device=device)

    with torch.no_grad():
        warm = layer(x).reshape(batch * tokens, projected).contiguous()
        _ = torch.cdist(warm[:topology_points], warm[:topology_points]).mean()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        start.record()
        activations = layer(x).reshape(batch * tokens, projected).contiguous()
        selected = activations[:topology_points]
        distances = torch.cdist(selected, selected)
        checksum = float(distances.mean().detach().cpu().item())
        end.record()
    torch.cuda.synchronize()
    cuda_ms = float(start.elapsed_time(end))

    cloud = selected.detach().cpu().numpy().astype(np.float64, copy=False)
    cpu_start = time.perf_counter()
    diagram = topoml.persistent_homology(cloud, max_dim=max_dim, max_radius=max_radius)
    features = topoml.PHFeaturizer(max_dim=max_dim, radii=[0.0, max_radius]).fit_transform([cloud])
    signature = topoml.activation_signature(cloud, radii=[0.0, max_radius], max_dim=max_dim)
    cpu_topology_ms = (time.perf_counter() - cpu_start) * 1000.0

    return {
        "benchmark": "cuda_tensor_topology",
        "claim_scope": "real CUDA tensor source and GPU distance timing; current persistent homology reduction runs on CPU",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "dtype": str(activations.dtype),
        "seed": seed,
        "shape": {
            "batch": batch,
            "tokens": tokens,
            "hidden": hidden,
            "projected": projected,
            "topology_points": topology_points,
        },
        "cuda_ms": cuda_ms,
        "cpu_topology_ms": cpu_topology_ms,
        "distance_checksum": checksum,
        "pairs": len(diagram.pairs),
        "feature_shape": list(features.shape),
        "signature": signature.values,
        "cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/cuda-tensor-topology.json"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--projected", type=int, default=8)
    parser.add_argument("--topology-points", type=int, default=32)
    parser.add_argument("--max-dim", type=int, default=1)
    parser.add_argument("--max-radius", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    result = run_cuda_tensor_case(
        batch=args.batch,
        tokens=args.tokens,
        hidden=args.hidden,
        projected=args.projected,
        topology_points=args.topology_points,
        max_dim=args.max_dim,
        max_radius=args.max_radius,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
