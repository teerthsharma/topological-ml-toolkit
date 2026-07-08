import numpy as np
import pytest

import topoml


def test_cpp_native_backend_compiles_and_matches_numpy(tmp_path):
    try:
        build = topoml.build_cpp_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = topoml.load_cpp_native_backend(build.library)
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 8.0],
        ],
        dtype=np.float64,
    )

    distances = backend.pairwise_l2(points)
    expected = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    edges = backend.threshold_edges(distances, radius=5.0)

    assert build.library.exists()
    assert np.allclose(distances, expected)
    assert edges.tolist() == [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]


def test_cpp_native_backend_helpers_are_exported():
    assert "build_cpp_native_backend" in topoml.__all__
    assert "load_cpp_native_backend" in topoml.__all__
