import math

import numpy as np
import pytest

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


def _topoml_deaths(points: np.ndarray, dim: int, max_dim: int = 1, max_radius: float = 10.0) -> np.ndarray:
    diagram = topoml.persistent_homology(points, max_dim=max_dim, max_radius=max_radius)
    return np.array(sorted(pair.death for pair in diagram.finite_pairs(dim) if pair.death is not None), dtype=float)


def _betti_from_intervals(intervals: np.ndarray, radius: float) -> int:
    count = 0
    for birth, death in intervals:
        if birth <= radius and (math.isinf(float(death)) or radius < death):
            count += 1
    return count


def test_h0_deaths_match_ripser_on_cluster_fixture():
    ripser_mod = pytest.importorskip("ripser")
    points = _cluster_fixture()

    result = ripser_mod.ripser(points, maxdim=0, thresh=10.0)
    ripser_deaths = np.array(sorted(float(death) for _birth, death in result["dgms"][0] if np.isfinite(death)))

    assert np.allclose(_topoml_deaths(points, dim=0, max_dim=0), ripser_deaths)
    assert np.allclose(ripser_deaths, np.array([0.2, 4.8]))


def test_h0_deaths_match_gudhi_on_cluster_fixture():
    gudhi = pytest.importorskip("gudhi")
    points = _cluster_fixture()

    complex_ = gudhi.RipsComplex(points=points.tolist(), max_edge_length=10.0)
    simplex_tree = complex_.create_simplex_tree(max_dimension=1)
    simplex_tree.persistence()
    intervals = simplex_tree.persistence_intervals_in_dimension(0)
    gudhi_deaths = np.array(sorted(float(death) for _birth, death in intervals if np.isfinite(death)))

    assert np.allclose(_topoml_deaths(points, dim=0, max_dim=0), gudhi_deaths)
    assert np.allclose(gudhi_deaths, np.array([0.2, 4.8]))


def test_h1_betti_count_matches_ripser_on_square_fixture():
    ripser_mod = pytest.importorskip("ripser")
    points = _square_fixture()

    diagram = topoml.persistent_homology(points, max_dim=1, max_radius=2.0)
    ripser_intervals = ripser_mod.ripser(points, maxdim=1, thresh=2.0)["dgms"][1]

    assert diagram.betti_at(1.1).beta1 == _betti_from_intervals(ripser_intervals, 1.1) == 1
    assert diagram.betti_at(1.5).beta1 == _betti_from_intervals(ripser_intervals, 1.5) == 0


def test_h1_betti_count_matches_gudhi_on_square_fixture():
    gudhi = pytest.importorskip("gudhi")
    points = _square_fixture()

    diagram = topoml.persistent_homology(points, max_dim=1, max_radius=2.0)
    complex_ = gudhi.RipsComplex(points=points.tolist(), max_edge_length=2.0)
    simplex_tree = complex_.create_simplex_tree(max_dimension=2)
    simplex_tree.persistence()
    gudhi_intervals = simplex_tree.persistence_intervals_in_dimension(1)

    assert diagram.betti_at(1.1).beta1 == _betti_from_intervals(gudhi_intervals, 1.1) == 1
    assert diagram.betti_at(1.5).beta1 == _betti_from_intervals(gudhi_intervals, 1.5) == 0
