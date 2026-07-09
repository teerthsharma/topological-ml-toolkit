from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    points: np.ndarray
    labels: np.ndarray
    expected_betti: dict[str, int]
    metadata: dict[str, Any]


def make_noisy_circle(
    *,
    n_samples: int = 64,
    radius: float = 1.0,
    noise: float = 0.03,
    random_state: int | None = 0,
) -> BenchmarkDataset:
    _validate_positive_int(n_samples, "n_samples")
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("radius must be a finite positive value")
    if noise < 0.0 or not np.isfinite(noise):
        raise ValueError("noise must be a finite non-negative value")
    rng = np.random.default_rng(random_state)
    angles = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    if noise > 0.0:
        points = points + rng.normal(scale=noise, size=points.shape)
    return BenchmarkDataset(
        name="noisy_circle",
        points=np.asarray(points, dtype=float),
        labels=np.zeros(n_samples, dtype=int),
        expected_betti={"beta0": 1, "beta1": 1},
        metadata={"radius": float(radius), "noise": float(noise), "random_state": random_state},
    )


def make_two_circles(
    *,
    n_samples: int = 64,
    radius: float = 1.0,
    separation: float = 3.0,
    noise: float = 0.03,
    random_state: int | None = 0,
) -> BenchmarkDataset:
    _validate_positive_int(n_samples, "n_samples")
    if separation <= 2.0 * radius:
        raise ValueError("separation must be greater than two radii")
    left = make_noisy_circle(n_samples=n_samples, radius=radius, noise=noise, random_state=random_state)
    right_seed = None if random_state is None else int(random_state) + 1
    right = make_noisy_circle(n_samples=n_samples, radius=radius, noise=noise, random_state=right_seed)
    points = np.vstack(
        [
            left.points + np.array([-separation / 2.0, 0.0]),
            right.points + np.array([separation / 2.0, 0.0]),
        ]
    )
    labels = np.asarray(["left"] * n_samples + ["right"] * n_samples, dtype=object)
    return BenchmarkDataset(
        name="two_circles",
        points=points,
        labels=labels,
        expected_betti={"components": 2, "loops": 2},
        metadata={
            "radius": float(radius),
            "separation": float(separation),
            "noise": float(noise),
            "random_state": random_state,
        },
    )


def make_cluster_bridge(*, random_state: int | None = 0) -> BenchmarkDataset:
    del random_state
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)
    labels = np.asarray(["near", "near", "far"], dtype=object)
    return BenchmarkDataset(
        name="cluster_bridge",
        points=points,
        labels=labels,
        expected_betti={"beta0@0.1": 3, "beta0@0.3": 2, "beta0@6": 1},
        metadata={"description": "three-point H0 merge fixture"},
    )


def list_benchmark_datasets() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_FACTORIES))


def load_benchmark_dataset(name: str, **kwargs: Any) -> BenchmarkDataset:
    normalized = name.strip().lower().replace("-", "_")
    try:
        factory = _DATASET_FACTORIES[normalized]
    except KeyError as exc:
        available = ", ".join(list_benchmark_datasets())
        raise KeyError(f"unknown benchmark dataset {name!r}; available: {available}") from exc
    return factory(**kwargs)


def _validate_positive_int(value: int, name: str) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


_DATASET_FACTORIES: dict[str, Callable[..., BenchmarkDataset]] = {
    "cluster_bridge": make_cluster_bridge,
    "noisy_circle": make_noisy_circle,
    "two_circles": make_two_circles,
}
