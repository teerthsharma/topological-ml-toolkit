from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import topoml


def test_rescale_persistence_diagram_preserves_essential_intervals() -> None:
    diagram = topoml.PersistenceDiagram(
        [
            topoml.PersistencePair(0, 0.0, 2.0),
            topoml.PersistencePair(0, 0.0, None),
            topoml.PersistencePair(1, 1.0, 3.0),
        ]
    )

    rescale = getattr(topoml, "rescale_persistence_diagram", None)
    assert rescale is not None, "public rescale_persistence_diagram API is missing"
    scaled = rescale(diagram, 0.25)

    assert scaled.pairs == (
        topoml.PersistencePair(0, 0.0, 0.5),
        topoml.PersistencePair(0, 0.0, None),
        topoml.PersistencePair(1, 0.25, 0.75),
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_rescale_persistence_diagram_rejects_invalid_scales(scale: float) -> None:
    diagram = topoml.PersistenceDiagram([topoml.PersistencePair(0, 0.0, None)])

    rescale = getattr(topoml, "rescale_persistence_diagram", None)
    assert rescale is not None, "public rescale_persistence_diagram API is missing"
    with pytest.raises(ValueError, match="positive finite"):
        rescale(diagram, scale)


def _rotation(theta: float) -> np.ndarray:
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, -sine], [sine, cosine]])


def _assert_diagrams_close(
    observed: tuple[topoml.PersistenceDiagram, ...],
    expected: tuple[topoml.PersistenceDiagram, ...],
) -> None:
    assert len(observed) == len(expected)
    for got_diagram, want_diagram in zip(observed, expected):
        assert len(got_diagram.pairs) == len(want_diagram.pairs)
        for got, want in zip(got_diagram.pairs, want_diagram.pairs):
            assert got.dimension == want.dimension
            assert got.birth == pytest.approx(want.birth, rel=1e-10, abs=1e-12)
            if want.death is None:
                assert got.death is None
            else:
                assert got.death == pytest.approx(want.death, rel=1e-10, abs=1e-12)


def test_similarity_trajectory_reuses_one_diagram_and_matches_dense() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    scales = np.array([1.0, 0.8, 0.55])
    trajectory = np.stack(
        [
            scale * (base @ _rotation(0.4 * index).T)
            + np.array([2.0 * index, -0.5 * index])
            for index, scale in enumerate(scales)
        ]
    )

    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    result = compute(trajectory, max_dim=1, tolerance=1e-10)
    dense = tuple(topoml.persistent_homology(frame, max_dim=1) for frame in trajectory)

    assert result.mode == "similarity-reuse"
    assert result.persistence_evaluations == 1
    assert result.certificate.certified
    np.testing.assert_allclose(result.certificate.scales, scales, rtol=1e-10, atol=1e-12)
    assert max(result.certificate.max_relative_distortion) <= 1e-10
    _assert_diagrams_close(result.diagrams, dense)


def test_similarity_diagram_obeys_betti_reparameterization() -> None:
    base = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]])
    scale = 0.4
    trajectory = np.stack([base, scale * base])

    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    result = compute(trajectory, max_dim=0)

    for radius in np.linspace(0.0, 5.0, 31):
        assert result.diagrams[1].betti_at(radius) == result.diagrams[0].betti_at(radius / scale)


@pytest.mark.parametrize("motion", ["anisotropic", "nonlinear", "independent"])
def test_non_similarity_motion_uses_dense_fallback(motion: str) -> None:
    base = np.random.default_rng(31).normal(size=(12, 2))
    if motion == "anisotropic":
        moved = base * np.array([0.7, 1.1])
    elif motion == "nonlinear":
        moved = np.sign(base) * np.square(base)
    else:
        moved = base.copy()
        moved[0] += np.array([0.3, -0.2])

    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    result = compute(np.stack([base, moved]), max_dim=0, tolerance=1e-10)

    assert not result.certificate.certified
    assert result.mode == "dense-fallback"
    assert result.persistence_evaluations == 2


def test_duplicate_base_points_that_separate_are_not_certified() -> None:
    base = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    moved = np.array([[0.0, 0.2], [0.0, -0.2], [np.sqrt(0.96), 0.0]])

    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    result = compute(np.stack([base, moved]), max_dim=0, tolerance=1e-10)

    assert not result.certificate.certified
    assert result.mode == "dense-fallback"
    assert result.diagrams[1].pairs != topoml.rescale_persistence_diagram(result.diagrams[0], 1.0).pairs


def test_finite_radius_uses_dense_fallback_to_preserve_cutoff_semantics() -> None:
    base = np.random.default_rng(77).normal(size=(10, 2))
    trajectory = np.stack([base, 0.4 * base])

    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    result = compute(trajectory, max_dim=0, max_radius=1.0)
    dense = tuple(
        topoml.persistent_homology(frame, max_dim=0, max_radius=1.0)
        for frame in trajectory
    )

    assert result.certificate.certified
    assert result.mode == "dense-fallback-finite-radius"
    assert result.persistence_evaluations == 2
    _assert_diagrams_close(result.diagrams, dense)


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        (np.array([]), "non-empty"),
        (np.zeros((2, 3)), "non-empty"),
        (np.zeros((1, 1, 2)), "at least two"),
        (np.array([[[0.0], [np.nan]]]), "finite"),
    ],
)
def test_similarity_trajectory_rejects_malformed_inputs(
    trajectory: np.ndarray,
    message: str,
) -> None:
    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    with pytest.raises(ValueError, match=message):
        compute(trajectory)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_similarity_trajectory_rejects_non_positive_or_non_finite_tolerance(
    tolerance: float,
) -> None:
    trajectory = np.zeros((1, 2, 1))
    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"
    with pytest.raises(ValueError, match="positive finite"):
        compute(trajectory, tolerance=tolerance)


def test_degenerate_base_uses_dense_fallback() -> None:
    trajectory = np.zeros((2, 3, 2))
    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"

    result = compute(trajectory, max_dim=0)

    assert not result.certificate.certified
    assert result.certificate.reason == "degenerate-base"
    assert result.mode == "dense-fallback"


def test_similarity_trajectory_rejects_overflowed_pairwise_distances() -> None:
    trajectory = np.array([[[1e308], [-1e308]]])
    compute = getattr(topoml, "persistence_similarity_trajectory", None)
    assert compute is not None, "public persistence_similarity_trajectory API is missing"

    with pytest.raises(ValueError, match="pairwise distances must be finite"):
        compute(trajectory, max_dim=0)


def test_similarity_benchmark_emits_raw_paired_evidence(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "persistence-similarity.json"

    subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_persistence_similarity.py",
            "--out",
            str(output),
            "--points",
            "8",
            "--frames",
            "3",
            "--repetitions",
            "2",
            "--warmups",
            "0",
            "--bootstrap-resamples",
            "100",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["claim_scope"] == "Python reference H0 on exact synthetic similarity trajectories"
    assert payload["cells"][0]["correctness"] == "pass"
    assert payload["cells"][0]["dense_persistence_evaluations"] == 3
    assert payload["cells"][0]["reuse_persistence_evaluations"] == 1
    assert payload["cells"][0]["work_reduction"] == pytest.approx(2.0 / 3.0)
    assert len(payload["cells"][0]["samples"]) == 2
