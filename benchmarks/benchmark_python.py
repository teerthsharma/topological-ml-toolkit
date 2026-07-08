from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import topoml


def run_case(n: int) -> dict:
    rng = np.random.default_rng(7)
    points = rng.normal(size=(n, 2))
    start = time.perf_counter()
    diagram = topoml.persistent_homology(points, max_dim=1, max_radius=1.0)
    elapsed = time.perf_counter() - start
    return {
        "backend": "python-reference",
        "points": n,
        "max_dim": 1,
        "max_radius": 1.0,
        "seconds": elapsed,
        "pairs": len(diagram.pairs),
        "python": platform.python_version(),
        "seed": 7,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/python-ph.json"))
    parser.add_argument("--points", type=int, nargs="+", default=[16, 32])
    args = parser.parse_args()

    results = [run_case(n) for n in args.points]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

