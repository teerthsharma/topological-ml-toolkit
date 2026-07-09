from __future__ import annotations

import numpy as np
import pytest

import topoml


def test_asm_native_backend_compiles_probes_and_matches_numpy(tmp_path):
    try:
        build = topoml.build_asm_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = topoml.load_asm_native_backend(build.library)
    features = backend.cpu_features()

    assert isinstance(features.leaf7_ebx, int)
    assert isinstance(features.xcr0, int)
    assert features.avx2 == bool((features.leaf7_ebx >> 5) & 1)
    assert features.avx512f == bool((features.leaf7_ebx >> 16) & 1)
    assert features.avx512f_os == (
        features.avx512f and (features.xcr0 & 0xE6) == 0xE6
    )
    assert backend.has_symbol("topoml_l2_sq_f32_avx512")

    left = np.array([1.0, -2.0, 3.5, 8.0], dtype=np.float32)
    right = np.array([0.0, 1.0, 1.5, -1.0], dtype=np.float32)
    observed = backend.l2_sq_f32(left, right)
    expected = float(np.sum((left - right) ** 2, dtype=np.float32))

    assert np.isclose(observed, expected, rtol=1e-6, atol=1e-6)

    dispatched = backend.l2_sq_f32_dispatched(left, right)
    assert np.isclose(dispatched.value, expected, rtol=1e-6, atol=1e-6)
    assert dispatched.backend in {"asm-scalar", "asm-avx512"}
    assert dispatched.used_avx512 == features.avx512f_os


def test_asm_native_backend_dispatch_matches_numpy_for_vector_lengths(tmp_path):
    try:
        build = topoml.build_asm_native_backend(build_dir=tmp_path)
    except topoml.NativeBackendUnavailable as exc:
        pytest.skip(str(exc))

    backend = topoml.load_asm_native_backend(build.library)
    rng = np.random.default_rng(20260709)
    for dim in [1, 4, 15, 16, 17, 64, 65]:
        left = rng.normal(size=dim).astype(np.float32)
        right = rng.normal(size=dim).astype(np.float32)
        observed = backend.l2_sq_f32_dispatched(left, right)
        expected = float(np.sum((left - right) ** 2, dtype=np.float32))

        assert np.isclose(observed.value, expected, rtol=1e-5, atol=1e-5)


def test_asm_native_backend_helpers_are_exported():
    assert "AsmNativeBackend" in topoml.__all__
    assert "build_asm_native_backend" in topoml.__all__
    assert "load_asm_native_backend" in topoml.__all__
