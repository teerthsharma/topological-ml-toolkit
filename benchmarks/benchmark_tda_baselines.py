from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path
from typing import Callable

import numpy as np

import topoml


def _cluster_fixture() -> np.ndarray:
    return np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)


def _square_fixture() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _timed(fn: Callable[[], dict]) -> tuple[dict, float]:
    start = time.perf_counter()
    payload = fn()
    elapsed = time.perf_counter() - start
    return payload, elapsed


def _betti_from_intervals(intervals: np.ndarray, radius: float) -> int:
    count = 0
    for birth, death in intervals:
        if birth <= radius and (math.isinf(float(death)) or radius < death):
            count += 1
    return count


def _topoml_cluster() -> dict:
    diagram = topoml.persistent_homology(_cluster_fixture(), max_dim=0, max_radius=10.0)
    return {
        "h0_deaths": sorted(float(pair.death) for pair in diagram.finite_pairs(0) if pair.death is not None),
        "pairs": len(diagram.pairs),
    }


def _topoml_square() -> dict:
    diagram = topoml.persistent_homology(_square_fixture(), max_dim=1, max_radius=2.0)
    return {
        "beta1_at_1_1": diagram.betti_at(1.1).beta1,
        "beta1_at_1_5": diagram.betti_at(1.5).beta1,
        "pairs": len(diagram.pairs),
    }


def _ripser_cluster() -> dict:
    from ripser import ripser

    result = ripser(_cluster_fixture(), maxdim=0, thresh=10.0)
    deaths = sorted(float(death) for _birth, death in result["dgms"][0] if np.isfinite(death))
    return {"h0_deaths": deaths, "pairs": int(result["dgms"][0].shape[0])}


def _ripser_square() -> dict:
    from ripser import ripser

    intervals = ripser(_square_fixture(), maxdim=1, thresh=2.0)["dgms"][1]
    return {
        "beta1_at_1_1": _betti_from_intervals(intervals, 1.1),
        "beta1_at_1_5": _betti_from_intervals(intervals, 1.5),
        "pairs": int(intervals.shape[0]),
    }


def _gudhi_cluster() -> dict:
    import gudhi

    complex_ = gudhi.RipsComplex(points=_cluster_fixture().tolist(), max_edge_length=10.0)
    simplex_tree = complex_.create_simplex_tree(max_dimension=1)
    simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    deaths = sorted(float(death) for _birth, death in intervals if np.isfinite(death))
    return {"h0_deaths": deaths, "pairs": int(intervals.shape[0])}


def _gudhi_square() -> dict:
    import gudhi

    complex_ = gudhi.RipsComplex(points=_square_fixture().tolist(), max_edge_length=2.0)
    simplex_tree = complex_.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(1)
    return {
        "beta1_at_1_1": _betti_from_intervals(intervals, 1.1),
        "beta1_at_1_5": _betti_from_intervals(intervals, 1.5),
        "pairs": int(intervals.shape[0]),
    }


def run() -> dict:
    rows: list[dict] = []
    for backend, fixture, fn in (
        ("topoml-python", "three-point-h0", _topoml_cluster),
        ("ripser", "three-point-h0", _ripser_cluster),
        ("gudhi", "three-point-h0", _gudhi_cluster),
        ("topoml-python", "unit-square-h1", _topoml_square),
        ("ripser", "unit-square-h1", _ripser_square),
        ("gudhi", "unit-square-h1", _gudhi_square),
    ):
        payload, seconds = _timed(fn)
        rows.append({"backend": backend, "fixture": fixture, "seconds": seconds, **payload})

    cluster_values = [row["h0_deaths"] for row in rows if row["fixture"] == "three-point-h0"]
    square_values = {
        (row["beta1_at_1_1"], row["beta1_at_1_5"]) for row in rows if row["fixture"] == "unit-square-h1"
    }
    return {
        "benchmark": "tda-baseline-parity",
        "claim_scope": "small deterministic Vietoris-Rips parity fixtures, not performance leadership",
        "python": platform.python_version(),
        "versions": {
            "topological-ml-toolkit": _version("topological-ml-toolkit"),
            "ripser": _version("ripser"),
            "gudhi": _version("gudhi"),
            "numpy": np.__version__,
        },
        "fixtures": {
            "three-point-h0": "finite H0 deaths must match [0.2, 4.8]",
            "unit-square-h1": "beta1 must be 1 at radius 1.1 and 0 at radius 1.5",
        },
        "parity": {
            "h0_deaths_match": all(np.allclose(cluster_values[0], values) for values in cluster_values[1:]),
            "h1_betti_counts_match": len(square_values) == 1,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/tda-baselines.json"))
    args = parser.parse_args()

    payload = run()
    if not all(payload["parity"].values()):
        raise SystemExit(f"TDA baseline parity failed: {payload['parity']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
