from __future__ import annotations

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def test_cuda_and_triton_sources_expose_expected_symbols():
    cuda_distance = (ROOT / "backends" / "cuda" / "topology_distance.cu").read_text(encoding="utf-8")
    cuda_reductions = (ROOT / "backends" / "cuda" / "warp_reductions.cu").read_text(encoding="utf-8")
    triton_distance = (ROOT / "backends" / "triton" / "topology_distance.py").read_text(encoding="utf-8")

    assert "topoml_pairwise_l2_f32" in cuda_distance
    assert "topoml_threshold_edges_u8" in cuda_distance
    assert "topoml_row_sum_f32" in cuda_reductions
    assert "topoml_persistence_image_accumulate_f32" in cuda_reductions
    assert "@triton.jit" in triton_distance
    assert "def pairwise_l2" in triton_distance


def test_pairwise_l2_fixture_matches_cuda_and_triton_contract():
    points = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]], dtype=np.float32)

    observed = _pairwise_l2(points)
    expected = np.array(
        [
            [0.0, 5.0, 10.0],
            [5.0, 0.0, 5.0],
            [10.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_threshold_edges_fixture_excludes_diagonal_and_uses_radius():
    distances = np.array(
        [
            [0.0, 0.2, 1.5],
            [0.2, 0.0, 0.9],
            [1.5, 0.9, 0.0],
        ],
        dtype=np.float32,
    )

    observed = _threshold_edges(distances, radius=0.9)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(observed, expected)


def test_row_sum_fixture_matches_cuda_reduction_contract():
    matrix = np.array([[1.0, 2.5, -0.5], [0.25, 0.75, 4.0]], dtype=np.float32)

    observed = _row_sum(matrix)
    expected = np.array([3.0, 5.0], dtype=np.float32)

    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-6)


def test_persistence_image_accumulation_fixture_matches_cuda_contract():
    births = np.array([0.0, 1.0], dtype=np.float32)
    persistences = np.array([1.0, 2.0], dtype=np.float32)

    observed = _persistence_image(
        births,
        persistences,
        width=2,
        height=2,
        birth_min=0.0,
        birth_max=1.0,
        persistence_min=1.0,
        persistence_max=2.0,
        sigma=1.0,
    )
    expected = np.array(
        [
            1.0 + 2.0 * np.exp(-1.0),
            np.exp(-0.5) + 2.0 * np.exp(-0.5),
            np.exp(-0.5) + 2.0 * np.exp(-0.5),
            np.exp(-1.0) + 2.0,
        ],
        dtype=np.float32,
    ).reshape(2, 2)

    assert np.allclose(observed, expected, rtol=1e-6, atol=1e-6)


def _pairwise_l2(points: np.ndarray) -> np.ndarray:
    deltas = points[:, None, :] - points[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2, dtype=np.float32), dtype=np.float32)


def _threshold_edges(distances: np.ndarray, radius: float) -> np.ndarray:
    edges = (distances <= radius).astype(np.uint8)
    np.fill_diagonal(edges, 0)
    return edges


def _row_sum(matrix: np.ndarray) -> np.ndarray:
    return np.sum(matrix, axis=1, dtype=np.float32)


def _persistence_image(
    births: np.ndarray,
    persistences: np.ndarray,
    *,
    width: int,
    height: int,
    birth_min: float,
    birth_max: float,
    persistence_min: float,
    persistence_max: float,
    sigma: float,
) -> np.ndarray:
    image = np.zeros((height, width), dtype=np.float32)
    denom = np.float32(2.0 * sigma * sigma)
    for y_idx in range(height):
        py = persistence_min + (persistence_max - persistence_min) * y_idx / max(1, height - 1)
        for x_idx in range(width):
            bx = birth_min + (birth_max - birth_min) * x_idx / max(1, width - 1)
            value = np.float32(0.0)
            for birth, persistence in zip(births, persistences):
                db = np.float32(bx - birth)
                dp = np.float32(py - persistence)
                value += np.float32(persistence) * np.exp(-((db * db + dp * dp) / denom), dtype=np.float32)
            image[y_idx, x_idx] = value
    return image
