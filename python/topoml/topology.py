from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Mapping, Sequence

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


@dataclass(frozen=True)
class PathHomotopySignature:
    winding_number: int
    total_angle: float
    closed: bool
    endpoint_distance: float


@dataclass(frozen=True)
class StratumSignature:
    stratum_counts: dict[str, int]
    n_strata: int
    boundary_fraction: float


@dataclass(frozen=True)
class OrbitSignature:
    orbit_size: int
    stabilizer_count: int
    orbit_diameter: float
    quotient_representative: tuple[float, ...]


@dataclass(frozen=True)
class EquivarianceResidual:
    action_residuals: dict[str, float]
    max_residual: float
    mean_residual: float


@dataclass(frozen=True)
class ScottFixedPointResult:
    fixed_point: np.ndarray
    steps: int
    converged: bool
    residual: float


@dataclass(frozen=True)
class WeakConvergenceResidual:
    probe_residuals: np.ndarray
    max_residual: float
    mean_residual: float
    strong_residual: float


@dataclass(frozen=True)
class TopologySignature:
    kind: str
    values: dict[str, float]


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


def path_homotopy_signature(
    path: np.ndarray,
    *,
    basepoint: Sequence[float] = (0.0, 0.0),
    closed_tolerance: float = 1e-6,
) -> PathHomotopySignature:
    pts = _as_points(path)
    if pts.shape[1] != 2:
        raise ValueError("path_homotopy_signature currently expects 2D path points")
    if pts.shape[0] < 2:
        raise ValueError("path must contain at least two points")
    if closed_tolerance < 0 or not np.isfinite(closed_tolerance):
        raise ValueError("closed_tolerance must be a finite non-negative value")
    center = np.asarray(basepoint, dtype=float).reshape(-1)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("basepoint must be a finite 2D point")
    shifted = pts - center
    if np.any(np.linalg.norm(shifted, axis=1) <= closed_tolerance):
        raise ValueError("path must not pass through the basepoint")
    endpoint_distance = float(np.linalg.norm(pts[0] - pts[-1]))
    closed = endpoint_distance <= closed_tolerance
    angles = np.unwrap(np.arctan2(shifted[:, 1], shifted[:, 0]))
    total_angle = float(angles[-1] - angles[0])
    winding = int(np.rint(total_angle / (2.0 * np.pi))) if closed else 0
    return PathHomotopySignature(
        winding_number=winding,
        total_angle=total_angle,
        closed=closed,
        endpoint_distance=endpoint_distance,
    )


def activation_strata(
    activations: np.ndarray,
    *,
    threshold: float = 0.0,
    boundary_tolerance: float = 1e-9,
) -> StratumSignature:
    values = np.asarray(activations, dtype=float)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("activations must have a non-empty sample axis and feature axis")
    if not np.isfinite(values).all():
        raise ValueError("activations must contain only finite values")
    if boundary_tolerance < 0 or not np.isfinite(boundary_tolerance):
        raise ValueError("boundary_tolerance must be a finite non-negative value")
    samples = values.reshape(values.shape[0], -1)
    masks = samples > threshold
    stratum_counts: dict[str, int] = {}
    for mask in masks:
        key = "".join("1" if item else "0" for item in mask)
        stratum_counts[key] = stratum_counts.get(key, 0) + 1
    boundary_hits = np.count_nonzero(np.abs(samples - threshold) <= boundary_tolerance)
    return StratumSignature(
        stratum_counts=dict(sorted(stratum_counts.items())),
        n_strata=len(stratum_counts),
        boundary_fraction=float(boundary_hits / samples.size),
    )


def finite_orbit_signature(
    point: np.ndarray,
    transforms: Sequence[np.ndarray],
    *,
    tolerance: float = 1e-9,
) -> OrbitSignature:
    value = np.asarray(point, dtype=float).reshape(-1)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("point must be a non-empty finite vector")
    if not transforms:
        raise ValueError("transforms must be non-empty")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")

    orbit: list[np.ndarray] = []
    stabilizer_count = 0
    for transform in transforms:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (value.size, value.size) or not np.isfinite(matrix).all():
            raise ValueError("each transform must be a finite square matrix matching point dimension")
        image = matrix @ value
        if np.linalg.norm(image - value) <= tolerance:
            stabilizer_count += 1
        if not any(np.linalg.norm(image - existing) <= tolerance for existing in orbit):
            orbit.append(image)

    if len(orbit) == 1:
        diameter = 0.0
    else:
        stacked = np.vstack(orbit)
        diameter = float(np.max(_pairwise_distances(stacked)))
    representative = tuple(float(x) for x in min((tuple(image.tolist()) for image in orbit)))
    return OrbitSignature(
        orbit_size=len(orbit),
        stabilizer_count=stabilizer_count,
        orbit_diameter=diameter,
        quotient_representative=representative,
    )


def equivariance_residual(
    points: np.ndarray,
    model: Callable[[np.ndarray], np.ndarray],
    input_actions: Mapping[str, np.ndarray],
    output_actions: Mapping[str, np.ndarray] | None = None,
) -> EquivarianceResidual:
    pts = _as_points(points)
    if not input_actions:
        raise ValueError("input_actions must be non-empty")
    baseline = _as_model_output(model(pts))
    residuals: dict[str, float] = {}
    for name, action in input_actions.items():
        matrix = np.asarray(action, dtype=float)
        if matrix.shape != (pts.shape[1], pts.shape[1]) or not np.isfinite(matrix).all():
            raise ValueError("each input action must be a finite square matrix matching point dimension")
        transformed_input = pts @ matrix.T
        observed = _as_model_output(model(transformed_input))
        if observed.shape != baseline.shape:
            raise ValueError("model output shape must be stable across actions")
        if output_actions is None:
            expected = baseline
        else:
            if name not in output_actions:
                raise KeyError(f"missing output action for {name!r}")
            output_matrix = np.asarray(output_actions[name], dtype=float)
            if output_matrix.shape != (baseline.shape[1], baseline.shape[1]) or not np.isfinite(output_matrix).all():
                raise ValueError("each output action must be a finite square matrix matching model output dimension")
            expected = baseline @ output_matrix.T
        residuals[name] = float(np.linalg.norm(observed - expected))
    values = tuple(residuals.values())
    return EquivarianceResidual(
        action_residuals=dict(sorted(residuals.items())),
        max_residual=max(values),
        mean_residual=float(np.mean(values)),
    )


def scott_fixed_point(
    update: Callable[[np.ndarray], np.ndarray],
    bottom: np.ndarray,
    *,
    join: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.maximum,
    max_steps: int = 64,
    tolerance: float = 0.0,
) -> ScottFixedPointResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be a finite non-negative value")
    current = np.asarray(bottom).copy()
    if current.size == 0:
        raise ValueError("bottom must be non-empty")
    last_residual = float("inf")
    for step in range(1, max_steps + 1):
        proposed = np.asarray(update(current.copy()))
        if proposed.shape != current.shape:
            raise ValueError("update output shape must match bottom")
        nxt = np.asarray(join(current, proposed))
        if nxt.shape != current.shape:
            raise ValueError("join output shape must match bottom")
        last_residual = _array_residual(current, nxt)
        follow = np.asarray(join(nxt, np.asarray(update(nxt.copy()))))
        if follow.shape != current.shape:
            raise ValueError("update/join follow-up shape must match bottom")
        if _array_residual(nxt, follow) <= tolerance:
            return ScottFixedPointResult(
                fixed_point=nxt,
                steps=step,
                converged=True,
                residual=float(_array_residual(nxt, follow)),
            )
        current = nxt
    return ScottFixedPointResult(
        fixed_point=current,
        steps=max_steps,
        converged=False,
        residual=last_residual,
    )


def weak_convergence_residual(
    sequence: np.ndarray,
    limit: np.ndarray,
    probes: np.ndarray,
) -> WeakConvergenceResidual:
    seq = np.asarray(sequence, dtype=float)
    target = np.asarray(limit, dtype=float).reshape(-1)
    probe_matrix = np.asarray(probes, dtype=float)
    if seq.ndim != 2 or seq.shape[0] == 0 or seq.shape[1] == 0:
        raise ValueError("sequence must be a non-empty 2D array")
    if target.shape != (seq.shape[1],):
        raise ValueError("limit must match sequence feature dimension")
    if probe_matrix.ndim != 2 or probe_matrix.shape[1] != seq.shape[1] or probe_matrix.shape[0] == 0:
        raise ValueError("probes must be a non-empty 2D array with matching feature dimension")
    if not (np.isfinite(seq).all() and np.isfinite(target).all() and np.isfinite(probe_matrix).all()):
        raise ValueError("sequence, limit, and probes must contain only finite values")
    delta = seq[-1] - target
    residuals = np.abs(probe_matrix @ delta)
    return WeakConvergenceResidual(
        probe_residuals=residuals,
        max_residual=float(np.max(residuals)),
        mean_residual=float(np.mean(residuals)),
        strong_residual=float(np.linalg.norm(delta)),
    )


def graph_signature(adjacency: np.ndarray) -> TopologySignature:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("adjacency must be a non-empty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("adjacency must contain only finite values")
    undirected = np.maximum(matrix, matrix.T) > 0
    np.fill_diagonal(undirected, False)
    n_nodes = undirected.shape[0]
    n_edges = int(np.count_nonzero(np.triu(undirected, k=1)))
    components = _graph_components(undirected)
    cycle_rank = n_edges - n_nodes + components
    return TopologySignature(
        kind="graph",
        values={
            "nodes": float(n_nodes),
            "edges": float(n_edges),
            "components": float(components),
            "cycle_rank": float(cycle_rank),
        },
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


def _graph_components(adjacency: np.ndarray) -> int:
    remaining = set(range(adjacency.shape[0]))
    components = 0
    while remaining:
        components += 1
        start = min(remaining)
        queue: deque[int] = deque([start])
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in np.flatnonzero(adjacency[current]):
                idx = int(neighbor)
                if idx in remaining:
                    remaining.remove(idx)
                    queue.append(idx)
    return components


def _as_model_output(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    if output.ndim == 1:
        output = output.reshape(-1, 1)
    if output.ndim != 2 or output.shape[0] == 0 or not np.isfinite(output).all():
        raise ValueError("model output must be a finite 1D or 2D array")
    return output


def _array_residual(left: np.ndarray, right: np.ndarray) -> float:
    if left.dtype == bool or right.dtype == bool:
        return 0.0 if np.array_equal(left, right) else float(np.count_nonzero(left != right))
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
