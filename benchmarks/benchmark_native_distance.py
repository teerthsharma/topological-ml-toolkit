from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import topoml


def run_case(n: int, dim: int, radius: float, build_dir: Path) -> dict:
    build = topoml.build_cpp_native_backend(build_dir=build_dir)
    backend = topoml.load_cpp_native_backend(build.library)
    rng = np.random.default_rng(11)
    points = rng.normal(size=(n, dim)).astype(np.float64)

    native_start = time.perf_counter()
    native_distances = backend.pairwise_l2(points)
    native_edges = backend.threshold_edges(native_distances, radius=radius)
    native_seconds = time.perf_counter() - native_start

    numpy_start = time.perf_counter()
    numpy_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    numpy_edges = ((numpy_distances <= radius) & (~np.eye(n, dtype=bool))).astype(np.uint8)
    numpy_seconds = time.perf_counter() - numpy_start

    if not np.allclose(native_distances, numpy_distances):
        raise AssertionError("native C++ distances differ from NumPy")
    if not np.array_equal(native_edges, numpy_edges):
        raise AssertionError("native C++ threshold edges differ from NumPy")

    return {
        "backend": "cpp-native-ctypes",
        "points": n,
        "dim": dim,
        "radius": radius,
        "native_seconds": native_seconds,
        "numpy_seconds": numpy_seconds,
        "edge_count": int(native_edges.sum()),
        "compiler": build.compiler,
        "library": str(build.library),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "claim_scope": "C++ preprocessing ABI correctness and timing smoke, not persistent-homology acceleration",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/native-distance.json"))
    parser.add_argument("--points", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--build-dir", type=Path, default=Path("artifacts/native"))
    args = parser.parse_args()

    results = [run_case(n, args.dim, args.radius, args.build_dir) for n in args.points]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
