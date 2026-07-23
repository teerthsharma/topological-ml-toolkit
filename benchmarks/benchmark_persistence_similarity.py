from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np

import topoml


CLAIM_SCOPE = "Python reference H0 on exact synthetic similarity trajectories"


def _rotation_z(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def _trajectory(points: np.ndarray, frame_count: int) -> np.ndarray:
    times = np.linspace(0.0, 1.0, frame_count)
    return np.stack(
        [
            np.exp(-0.15 * value) * (points @ _rotation_z(0.4 * value).T)
            + np.array([value, -2.0 * value, 0.5 * value])
            for value in times
        ]
    )


def _diagram_checksum(diagrams: tuple[topoml.PersistenceDiagram, ...]) -> float:
    return float(
        sum(
            pair.birth + (0.0 if pair.death is None else pair.death)
            for diagram in diagrams
            for pair in diagram.pairs
        )
    )


def _assert_diagrams_close(
    observed: tuple[topoml.PersistenceDiagram, ...],
    expected: tuple[topoml.PersistenceDiagram, ...],
) -> None:
    if len(observed) != len(expected):
        raise AssertionError("diagram count differs")
    for got_diagram, want_diagram in zip(observed, expected):
        if len(got_diagram.pairs) != len(want_diagram.pairs):
            raise AssertionError("persistence-pair count differs")
        for got, want in zip(got_diagram.pairs, want_diagram.pairs):
            if got.dimension != want.dimension or (got.death is None) != (want.death is None):
                raise AssertionError("persistence-pair type differs")
            if not np.isclose(got.birth, want.birth, rtol=1e-10, atol=1e-12):
                raise AssertionError("birth coordinate differs")
            if got.death is not None and not np.isclose(
                got.death,
                want.death,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise AssertionError("death coordinate differs")


def _timed(function) -> tuple[float, float]:
    start = time.perf_counter()
    checksum = function()
    return time.perf_counter() - start, checksum


def _bootstrap_speedup(
    samples: list[dict[str, float | int | str]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | list[float]]:
    log_ratios = np.log(
        np.asarray(
            [
                float(sample["dense_seconds"]) / float(sample["reuse_seconds"])
                for sample in samples
            ]
        )
    )
    geometric_mean = float(np.exp(np.mean(log_ratios)))
    if len(log_ratios) == 1:
        interval = [geometric_mean, geometric_mean]
    else:
        rng = np.random.default_rng(seed)
        draws = rng.choice(log_ratios, size=(resamples, len(log_ratios)), replace=True)
        bootstrapped = np.exp(np.mean(draws, axis=1))
        interval = [
            float(np.quantile(bootstrapped, 0.005)),
            float(np.quantile(bootstrapped, 0.995)),
        ]
    return {
        "geometric_mean_speedup": geometric_mean,
        "confidence": 0.99,
        "interval": interval,
    }


def _benchmark_cell(
    *,
    point_count: int,
    frame_count: int,
    repetitions: int,
    warmups: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    cloud = np.random.default_rng(seed).normal(size=(point_count, 3))
    frames = _trajectory(cloud, frame_count)

    def dense() -> float:
        diagrams = tuple(topoml.persistent_homology(frame, max_dim=0) for frame in frames)
        return _diagram_checksum(diagrams)

    def reuse() -> float:
        result = topoml.persistence_similarity_trajectory(frames, max_dim=0, tolerance=1e-10)
        if result.mode != "similarity-reuse":
            raise AssertionError(f"unexpected candidate mode: {result.mode}")
        return _diagram_checksum(result.diagrams)

    dense_diagrams = tuple(topoml.persistent_homology(frame, max_dim=0) for frame in frames)
    reuse_result = topoml.persistence_similarity_trajectory(frames, max_dim=0, tolerance=1e-10)
    _assert_diagrams_close(reuse_result.diagrams, dense_diagrams)

    for _ in range(warmups):
        dense()
        reuse()

    order_rng = random.Random(seed)
    samples: list[dict[str, float | int | str]] = []
    for repetition in range(repetitions):
        order = "dense-first" if order_rng.randrange(2) == 0 else "reuse-first"
        if order == "dense-first":
            dense_seconds, dense_checksum = _timed(dense)
            reuse_seconds, reuse_checksum = _timed(reuse)
        else:
            reuse_seconds, reuse_checksum = _timed(reuse)
            dense_seconds, dense_checksum = _timed(dense)
        if not math.isclose(dense_checksum, reuse_checksum, rel_tol=1e-10, abs_tol=1e-9):
            raise AssertionError("timed result checksums differ")
        samples.append(
            {
                "repetition": repetition,
                "order": order,
                "dense_seconds": dense_seconds,
                "reuse_seconds": reuse_seconds,
                "speedup": dense_seconds / reuse_seconds,
            }
        )

    return {
        "points": point_count,
        "frames": frame_count,
        "correctness": "pass",
        "max_relative_distortion": max(reuse_result.certificate.max_relative_distortion),
        "dense_persistence_evaluations": frame_count,
        "reuse_persistence_evaluations": reuse_result.persistence_evaluations,
        "work_reduction": 1.0 - reuse_result.persistence_evaluations / frame_count,
        "speedup": _bootstrap_speedup(
            samples,
            resamples=bootstrap_resamples,
            seed=seed + 17,
        ),
        "samples": samples,
    }


def run_benchmark(
    *,
    points: list[int],
    frames: list[int],
    repetitions: int,
    warmups: int,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    if any(value < 2 for value in points):
        raise ValueError("point counts must be at least two")
    if any(value < 1 for value in frames):
        raise ValueError("frame counts must be positive")
    if repetitions < 1 or warmups < 0 or bootstrap_resamples < 1:
        raise ValueError("repetitions and resamples must be positive; warmups must be non-negative")
    cells = [
        _benchmark_cell(
            point_count=point_count,
            frame_count=frame_count,
            repetitions=repetitions,
            warmups=warmups,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed + 1009 * point_count + frame_count,
        )
        for point_count in points
        for frame_count in frames
    ]
    return {
        "schema_version": 1,
        "claim_scope": CLAIM_SCOPE,
        "claim_boundary": (
            "Exact synthetic similarity paths and the Python reference backend only; "
            "not a universal backend speed claim."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "processor": platform.processor() or platform.machine(),
        },
        "configuration": {
            "seed": seed,
            "points": points,
            "frames": frames,
            "repetitions": repetitions,
            "warmups": warmups,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--frames", nargs="+", type=int, default=[8, 32])
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=3109)
    args = parser.parse_args()

    payload = run_benchmark(
        points=args.points,
        frames=args.frames,
        repetitions=args.repetitions,
        warmups=args.warmups,
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"claim_scope": payload["claim_scope"], "cells": len(payload["cells"])}))


if __name__ == "__main__":
    main()
