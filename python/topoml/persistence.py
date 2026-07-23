"""Certified persistence reuse for uniformly scaled point-cloud trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core import PersistenceDiagram, PersistencePair, persistent_homology


@dataclass(frozen=True)
class SimilarityCertificate:
    """Numerical evidence that every frame uniformly scales base distances."""

    certified: bool
    scales: tuple[float, ...]
    max_relative_distortion: tuple[float, ...]
    tolerance: float
    reason: str


@dataclass(frozen=True)
class PersistenceTrajectory:
    """Persistence diagrams plus the computation path used to produce them."""

    diagrams: tuple[PersistenceDiagram, ...]
    mode: str
    persistence_evaluations: int
    certificate: SimilarityCertificate


def rescale_persistence_diagram(
    diagram: PersistenceDiagram,
    scale: float,
) -> PersistenceDiagram:
    """Return a diagram with every finite filtration coordinate scaled."""
    value = float(scale)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("scale must be positive finite")
    return PersistenceDiagram(
        PersistencePair(
            dimension=pair.dimension,
            birth=pair.birth * value,
            death=None if pair.death is None else pair.death * value,
        )
        for pair in diagram.pairs
    )


def _pair_distances(points: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(points.shape[0], k=1)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(points[rows] - points[columns], axis=1)
    if not np.isfinite(distances).all():
        raise ValueError("pairwise distances must be finite")
    return distances


def _similarity_certificate(
    frames: np.ndarray,
    tolerance: float,
) -> SimilarityCertificate:
    base_distances = _pair_distances(frames[0])
    characteristic_distance = float(np.max(base_distances, initial=0.0))
    threshold = np.finfo(float).eps * max(1.0, characteristic_distance)
    usable = base_distances > threshold
    if not np.any(usable):
        frame_count = frames.shape[0]
        return SimilarityCertificate(
            certified=False,
            scales=tuple(float("nan") for _ in range(frame_count)),
            max_relative_distortion=tuple(float("inf") for _ in range(frame_count)),
            tolerance=tolerance,
            reason="degenerate-base",
        )

    nonzero_base_distances = base_distances[usable]
    denominator = float(np.dot(nonzero_base_distances, nonzero_base_distances))
    scales: list[float] = []
    distortions: list[float] = []
    certified = True
    for frame in frames:
        frame_distances = _pair_distances(frame)
        scale = float(
            np.dot(nonzero_base_distances, frame_distances[usable]) / denominator
        )
        if not math.isfinite(scale) or scale <= 0.0:
            scales.append(scale)
            distortions.append(float("inf"))
            certified = False
            continue

        residual = np.abs(frame_distances - scale * base_distances)
        zero_pair_scale = scale * max(characteristic_distance, threshold)
        normalizer = np.where(usable, scale * base_distances, zero_pair_scale)
        normalizer = np.maximum(normalizer, threshold)
        distortion = float(np.max(residual / normalizer, initial=0.0))
        scales.append(scale)
        distortions.append(distortion)
        if distortion > tolerance:
            certified = False

    return SimilarityCertificate(
        certified=certified,
        scales=tuple(scales),
        max_relative_distortion=tuple(distortions),
        tolerance=tolerance,
        reason="certified" if certified else "pair-distance-distortion",
    )


def persistence_similarity_trajectory(
    points: np.ndarray,
    max_dim: int = 1,
    max_radius: float = float("inf"),
    tolerance: float = 1e-10,
) -> PersistenceTrajectory:
    """Compute trajectory persistence with exact reuse when it is certified.

    A certified frame has pair distances ``d_t(i, j) = s_t d_0(i, j)`` for
    one positive ``s_t``. Its Vietoris--Rips persistence intervals are therefore
    the base intervals scaled by ``s_t``. Finite cutoffs deliberately use the
    dense path because the reference reducer's truncation changes which bars
    appear essential.
    """
    frames = np.asarray(points, dtype=float)
    if frames.ndim != 3 or frames.shape[0] == 0 or frames.shape[2] == 0:
        raise ValueError("points must be a non-empty [frame, point, coordinate] array")
    if frames.shape[1] < 2:
        raise ValueError("each frame must contain at least two points")
    if not np.isfinite(frames).all():
        raise ValueError("points must contain only finite values")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value <= 0.0:
        raise ValueError("tolerance must be positive finite")

    certificate = _similarity_certificate(frames, tolerance_value)
    finite_cutoff = math.isfinite(float(max_radius))
    if certificate.certified and not finite_cutoff:
        base_diagram = persistent_homology(
            frames[0],
            max_dim=max_dim,
            max_radius=max_radius,
        )
        diagrams = tuple(
            rescale_persistence_diagram(base_diagram, scale)
            for scale in certificate.scales
        )
        return PersistenceTrajectory(
            diagrams=diagrams,
            mode="similarity-reuse",
            persistence_evaluations=1,
            certificate=certificate,
        )

    diagrams = tuple(
        persistent_homology(frame, max_dim=max_dim, max_radius=max_radius)
        for frame in frames
    )
    mode = (
        "dense-fallback-finite-radius"
        if certificate.certified and finite_cutoff
        else "dense-fallback"
    )
    return PersistenceTrajectory(
        diagrams=diagrams,
        mode=mode,
        persistence_evaluations=len(diagrams),
        certificate=certificate,
    )
