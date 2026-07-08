from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BettiNumbers:
    beta0: int = 0
    beta1: int = 0
    beta2: int = 0


@dataclass(frozen=True)
class PersistencePair:
    dimension: int
    birth: float
    death: float | None


class PersistenceDiagram:
    def __init__(self, pairs: Iterable[PersistencePair]):
        self.pairs = tuple(
            sorted(
                pairs,
                key=lambda pair: (
                    pair.dimension,
                    pair.birth,
                    float("inf") if pair.death is None else pair.death,
                ),
            )
        )

    def betti_at(self, radius: float) -> BettiNumbers:
        beta0 = beta1 = beta2 = 0
        for pair in self.pairs:
            if pair.birth <= radius and (pair.death is None or radius < pair.death):
                if pair.dimension == 0:
                    beta0 += 1
                elif pair.dimension == 1:
                    beta1 += 1
                elif pair.dimension == 2:
                    beta2 += 1
        return BettiNumbers(beta0=beta0, beta1=beta1, beta2=beta2)

    def finite_pairs(self, dimension: int | None = None) -> list[PersistencePair]:
        return [
            pair
            for pair in self.pairs
            if pair.death is not None and (dimension is None or pair.dimension == dimension)
        ]

    def to_plotly_trace(self, dimension: int = 0) -> dict:
        finite = self.finite_pairs(dimension)
        return {
            "type": "scatter",
            "mode": "markers",
            "name": f"H{dimension} intervals",
            "x": [pair.birth for pair in finite],
            "y": [round(pair.death - pair.birth, 12) for pair in finite if pair.death is not None],
        }


def time_delay_embedding(samples: np.ndarray, dimension: int, tau: int = 1) -> np.ndarray:
    values = np.asarray(samples, dtype=float).reshape(-1)
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if tau <= 0:
        raise ValueError("tau must be positive")
    needed = (dimension - 1) * tau + 1
    if values.size < needed:
        raise ValueError("not enough samples for requested embedding")
    return np.array(
        [[values[start + dim * tau] for dim in range(dimension)] for start in range(values.size + 1 - needed)],
        dtype=float,
    )


def persistent_homology(
    points: np.ndarray,
    max_dim: int = 1,
    max_radius: float = float("inf"),
) -> PersistenceDiagram:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] == 0:
        raise ValueError("points must be a non-empty 2D array")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    if max_dim < 0 or max_dim > 2:
        raise ValueError("max_dim must be 0, 1, or 2")
    if np.isnan(max_radius) or max_radius < 0:
        raise ValueError("max_radius must be non-negative")

    simplices = _vietoris_rips_simplices(pts, max_dim=max_dim, max_radius=max_radius)
    return _reduce_z2(simplices, max_dim=max_dim)


def _vietoris_rips_simplices(points: np.ndarray, max_dim: int, max_radius: float) -> list[tuple[tuple[int, ...], float]]:
    n = points.shape[0]
    dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    simplices: list[tuple[tuple[int, ...], float]] = [((i,), 0.0) for i in range(n)]

    for size in range(2, max_dim + 3):
        for vertices in combinations(range(n), size):
            radius = max(dists[i, j] for i, j in combinations(vertices, 2))
            if radius <= max_radius:
                simplices.append((vertices, float(radius)))

    simplices.sort(key=lambda item: (item[1], len(item[0]) - 1, item[0]))
    return simplices


def _reduce_z2(simplices: list[tuple[tuple[int, ...], float]], max_dim: int) -> PersistenceDiagram:
    index = {vertices: idx for idx, (vertices, _filtration) in enumerate(simplices)}
    reduced_columns: list[list[int]] = []
    low_owner: dict[int, int] = {}
    paired_birth: set[int] = set()
    pairs: list[PersistencePair] = []

    for j, (vertices, _filtration) in enumerate(simplices):
        column = sorted(index[face] for face in _faces(vertices) if face in index)
        while column and column[-1] in low_owner:
            column = _xor_sorted(column, reduced_columns[low_owner[column[-1]]])
        if column:
            low = column[-1]
            low_owner[low] = j
            paired_birth.add(low)
            dimension = len(simplices[low][0]) - 1
            if dimension <= max_dim:
                pairs.append(
                    PersistencePair(
                        dimension=dimension,
                        birth=simplices[low][1],
                        death=simplices[j][1],
                    )
                )
        reduced_columns.append(column)

    for idx, column in enumerate(reduced_columns):
        dimension = len(simplices[idx][0]) - 1
        if not column and idx not in paired_birth and dimension <= max_dim:
            pairs.append(PersistencePair(dimension=dimension, birth=simplices[idx][1], death=None))
    return PersistenceDiagram(pairs)


def _faces(vertices: tuple[int, ...]) -> list[tuple[int, ...]]:
    if len(vertices) == 1:
        return []
    return [vertices[:i] + vertices[i + 1 :] for i in range(len(vertices))]


def _xor_sorted(left: list[int], right: list[int]) -> list[int]:
    out: list[int] = []
    i = j = 0
    while i < len(left) or j < len(right):
        if j == len(right) or (i < len(left) and left[i] < right[j]):
            out.append(left[i])
            i += 1
        elif i == len(left) or right[j] < left[i]:
            out.append(right[j])
            j += 1
        else:
            i += 1
            j += 1
    return out

