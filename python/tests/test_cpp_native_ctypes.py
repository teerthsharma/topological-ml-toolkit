import numpy as np
import pytest

import topoml


def _load_backend_or_skip(library):
    try:
        return topoml.load_cpp_native_backend(library)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))


def test_cpp_native_backend_compiles_and_matches_numpy(tmp_path):
    try:
        build = topoml.build_cpp_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = _load_backend_or_skip(build.library)
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


def test_cpp_native_backend_computes_h0_barcode_equivalent_to_python(tmp_path):
    try:
        build = topoml.build_cpp_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = _load_backend_or_skip(build.library)
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=np.float64)

    native = backend.persistent_homology_h0(points, max_radius=10.0)
    reference = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)

    assert [pair.dimension for pair in native.pairs] == [pair.dimension for pair in reference.pairs]
    assert [pair.birth for pair in native.pairs] == [pair.birth for pair in reference.pairs]
    assert [
        None if pair.death is None else round(pair.death, 10)
        for pair in native.pairs
    ] == [
        None if pair.death is None else round(pair.death, 10)
        for pair in reference.pairs
    ]
    assert native.betti_at(0.1).beta0 == 3
    assert native.betti_at(0.3).beta0 == 2
    assert native.betti_at(6.0).beta0 == 1


def test_cpp_backend_metadata_is_active_after_native_barcode_gate():
    metadata = {backend.id: backend for backend in topoml.available_backends()}

    assert metadata["cpp"].active
    assert metadata["cpp"].available
    assert not metadata["cpp"].planned
    assert "persistent_homology_h0" in metadata["cpp"].capabilities


def test_cpp_native_backend_helpers_are_exported():
    assert "build_cpp_native_backend" in topoml.__all__
    assert "load_cpp_native_backend" in topoml.__all__
