from __future__ import annotations

import numpy as np
import pytest

import topoml


def test_cuda_native_runtime_helpers_are_exported() -> None:
    assert "CudaBuildResult" in topoml.__all__
    assert "CudaNativeBackend" in topoml.__all__
    assert "build_cuda_native_backend" in topoml.__all__
    assert "load_cuda_native_backend" in topoml.__all__


def test_cuda_native_backend_compiles_and_matches_numpy_when_available(tmp_path) -> None:
    try:
        build = topoml.build_cuda_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = topoml.load_cuda_native_backend(build.library)
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [3.0, 4.0, 1.0],
            [6.0, 8.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    try:
        observed = backend.pairwise_l2(points)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    expected = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).astype(np.float32)
    assert np.allclose(observed, expected, rtol=1e-5, atol=1e-5)

    edges = backend.threshold_edges(observed, radius=5.0)
    expected_edges = ((expected <= 5.0) & (~np.eye(points.shape[0], dtype=bool))).astype(np.uint8)
    assert np.array_equal(edges, expected_edges)
