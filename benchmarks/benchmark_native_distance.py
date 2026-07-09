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

    native_h0_start = time.perf_counter()
    native_h0 = backend.persistent_homology_h0(points, max_radius=radius)
    native_h0_seconds = time.perf_counter() - native_h0_start

    numpy_start = time.perf_counter()
    numpy_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    numpy_edges = ((numpy_distances <= radius) & (~np.eye(n, dtype=bool))).astype(np.uint8)
    numpy_seconds = time.perf_counter() - numpy_start

    python_h0_start = time.perf_counter()
    python_h0 = topoml.persistent_homology(points, max_dim=0, max_radius=radius)
    python_h0_seconds = time.perf_counter() - python_h0_start

    if not np.allclose(native_distances, numpy_distances):
        raise AssertionError("native C++ distances differ from NumPy")
    if not np.array_equal(native_edges, numpy_edges):
        raise AssertionError("native C++ threshold edges differ from NumPy")
    native_deaths = sorted(pair.death for pair in native_h0.finite_pairs(0) if pair.death is not None)
    python_deaths = sorted(pair.death for pair in python_h0.finite_pairs(0) if pair.death is not None)
    if not np.allclose(native_deaths, python_deaths):
        raise AssertionError("native C++ H0 barcode deaths differ from Python reference")

    return {
        "backend": "cpp-native-ctypes",
        "points": n,
        "dim": dim,
        "radius": radius,
        "native_seconds": native_seconds,
        "numpy_seconds": numpy_seconds,
        "native_h0_seconds": native_h0_seconds,
        "python_h0_seconds": python_h0_seconds,
        "native_h0_pairs": len(native_h0.pairs),
        "edge_count": int(native_edges.sum()),
        "compiler": build.compiler,
        "library": str(build.library),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "claim_scope": "C++ preprocessing plus H0 barcode correctness and timing smoke, not H1/H2 persistent-homology acceleration",
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
