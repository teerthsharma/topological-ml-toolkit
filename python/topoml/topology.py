from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CoverCell:
    id: int
    center: int | None
    members: tuple[int, ...]
    interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class Cover:
    cells: tuple[CoverCell, ...]
    n_points: int
    radius: float | None = None
    kind: str = "metric"


@dataclass(frozen=True)
class NerveGraph:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MapperGraph:
    nodes: tuple[CoverCell, ...]
    edges: tuple[tuple[int, int], ...]
    intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SheafResidual:
    edge_residuals: dict[str, float]
    max_residual: float
    mean_residual: float


def metric_cover(points: np.ndarray, radius: float) -> Cover:
    pts = _as_points(points)
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("radius must be a finite non-negative value")

    dists = _pairwise_distances(pts)
    cells = []
    for idx in range(pts.shape[0]):
        members = tuple(int(member) for member in np.flatnonzero(dists[idx] <= radius))
        cells.append(CoverCell(id=idx, center=idx, members=members))
    return Cover(cells=tuple(cells), n_points=pts.shape[0], radius=radius, kind="metric")


def nerve_graph(cover: Cover) -> NerveGraph:
    edges = []
    for left, right in combinations(cover.cells, 2):
        if set(left.members).intersection(right.members):
            edges.append((left.id, right.id))
    return NerveGraph(
        nodes=tuple(cell.id for cell in cover.cells),
        edges=tuple(sorted(edges)),
    )


def mapper_graph(
    points: np.ndarray,
    filter_values: Sequence[float],
    *,
    intervals: int,
    overlap: float,
    cluster_radius: float,
) -> MapperGraph:
    pts = _as_points(points)
    filters = np.asarray(filter_values, dtype=float).reshape(-1)
    if filters.shape[0] != pts.shape[0]:
        raise ValueError("filter_values length must match number of points")
    if intervals <= 0:
        raise ValueError("intervals must be positive")
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must satisfy 0 <= overlap < 1")
    if cluster_radius < 0 or not np.isfinite(cluster_radius):
        raise ValueError("cluster_radius must be a finite non-negative value")

    interval_bounds = _overlapping_intervals(filters, intervals, overlap)
    dists = _pairwise_distances(pts)
    nodes: list[CoverCell] = []
    for interval in interval_bounds:
        members = tuple(int(idx) for idx in np.flatnonzero((filters >= interval[0]) & (filters <= interval[1])))
        for component in _connected_components(members, dists, cluster_radius):
            if component:
                nodes.append(
                    CoverCell(
                        id=len(nodes),
                        center=None,
                        members=component,
                        interval=interval,
                    )
                )

    edges = []
    for left, right in combinations(nodes, 2):
        if set(left.members).intersection(right.members):
            edges.append((left.id, right.id))
    return MapperGraph(nodes=tuple(nodes), edges=tuple(sorted(edges)), intervals=tuple(interval_bounds))


def sheaf_consistency_residual(
    sections: Mapping[str, np.ndarray],
    restrictions: Sequence[tuple[str, str, np.ndarray]],
) -> SheafResidual:
    edge_residuals: dict[str, float] = {}
    for source, target, matrix in restrictions:
        if source not in sections or target not in sections:
            raise KeyError(f"restriction references unknown section: {source!r}->{target!r}")
        source_values = np.asarray(sections[source], dtype=float).reshape(-1)
        target_values = np.asarray(sections[target], dtype=float).reshape(-1)
        restriction = np.asarray(matrix, dtype=float)
        projected = restriction @ source_values
        if projected.shape != target_values.shape:
            raise ValueError("restriction output shape must match target section shape")
        edge_residuals[f"{source}->{target}"] = float(np.linalg.norm(projected - target_values))

    values = tuple(edge_residuals.values())
    return SheafResidual(
        edge_residuals=edge_residuals,
        max_residual=max(values) if values else 0.0,
        mean_residual=float(np.mean(values)) if values else 0.0,
    )


def _as_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] == 0:
        raise ValueError("points must be a non-empty 2D array")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    return pts


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def _overlapping_intervals(values: np.ndarray, count: int, overlap: float) -> list[tuple[float, float]]:
    lo = float(values.min())
    hi = float(values.max())
    if lo == hi:
        return [(lo, hi)]
    width = (hi - lo) / (count - (count - 1) * overlap)
    step = width * (1.0 - overlap)
    return [(lo + idx * step, lo + idx * step + width) for idx in range(count)]


def _connected_components(members: tuple[int, ...], dists: np.ndarray, radius: float) -> list[tuple[int, ...]]:
    remaining = set(members)
    components: list[tuple[int, ...]] = []
    while remaining:
        start = min(remaining)
        queue: deque[int] = deque([start])
        remaining.remove(start)
        component = [start]
        while queue:
            current = queue.popleft()
            neighbors = [candidate for candidate in tuple(remaining) if dists[current, candidate] <= radius]
            for neighbor in neighbors:
                remaining.remove(neighbor)
                queue.append(neighbor)
                component.append(neighbor)
        components.append(tuple(sorted(component)))
    return components
