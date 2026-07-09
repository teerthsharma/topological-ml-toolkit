from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import PersistenceDiagram, PersistencePair


@dataclass(frozen=True)
class NativeBuildResult:
    source: Path
    library: Path
    compiler: str
    command: tuple[str, ...]


class NativeBackendUnavailable(RuntimeError):
    pass


class CppNativeBackend:
    """ctypes wrapper around the portable C++ native preprocessing ABI."""

    def __init__(self, library: Path):
        self.library = Path(library)
        try:
            self._lib = ctypes.CDLL(str(self.library))
        except OSError as exc:
            raise NativeBackendUnavailable(f"could not load C++ native library {self.library}: {exc}") from exc
        self._lib.topoml_pairwise_l2_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib.topoml_pairwise_l2_f64.restype = ctypes.c_int
        self._lib.topoml_threshold_edges_u8.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_double,
        ]
        self._lib.topoml_threshold_edges_u8.restype = ctypes.c_int
        self._lib.topoml_h0_barcode_f64.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_double,
        ]
        self._lib.topoml_h0_barcode_f64.restype = ctypes.c_int

    def pairwise_l2(self, points: np.ndarray) -> np.ndarray:
        pts = _as_points(points, dtype=np.float64)
        out = np.zeros((pts.shape[0], pts.shape[0]), dtype=np.float64)
        code = self._lib.topoml_pairwise_l2_f64(
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int64(pts.shape[0]),
            ctypes.c_int64(pts.shape[1]),
        )
        if code != 0:
            raise NativeBackendUnavailable(f"C++ pairwise distance returned error code {code}")
        return out

    def threshold_edges(self, distances: np.ndarray, radius: float) -> np.ndarray:
        dists = np.ascontiguousarray(distances, dtype=np.float64)
        if dists.ndim != 2 or dists.shape[0] != dists.shape[1] or dists.shape[0] == 0:
            raise ValueError("distances must be a non-empty square matrix")
        if radius < 0.0 or not np.isfinite(radius):
            raise ValueError("radius must be a finite non-negative value")
        edges = np.zeros(dists.shape, dtype=np.uint8)
        code = self._lib.topoml_threshold_edges_u8(
            dists.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            edges.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(dists.shape[0]),
            ctypes.c_double(float(radius)),
        )
        if code != 0:
            raise NativeBackendUnavailable(f"C++ threshold edges returned error code {code}")
        return edges

    def persistent_homology_h0(self, points: np.ndarray, max_radius: float = float("inf")) -> PersistenceDiagram:
        pts = _as_points(points, dtype=np.float64)
        if max_radius < 0.0 or np.isnan(max_radius):
            raise ValueError("max_radius must be non-negative")
        births = np.zeros(pts.shape[0], dtype=np.float64)
        deaths = np.zeros(pts.shape[0], dtype=np.float64)
        count = ctypes.c_int64(0)
        code = self._lib.topoml_h0_barcode_f64(
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            births.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            deaths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(count),
            ctypes.c_int64(pts.shape[0]),
            ctypes.c_int64(pts.shape[1]),
            ctypes.c_double(float(max_radius)),
        )
        if code != 0:
            raise NativeBackendUnavailable(f"C++ H0 barcode returned error code {code}")
        pairs = [
            PersistencePair(
                dimension=0,
                birth=float(births[index]),
                death=None if deaths[index] < 0.0 else float(deaths[index]),
            )
            for index in range(count.value)
        ]
        return PersistenceDiagram(pairs)


def build_cpp_native_backend(
    *,
    source: Path | None = None,
    build_dir: Path | None = None,
    compiler: str | None = None,
) -> NativeBuildResult:
    root = Path(__file__).resolve().parents[2]
    src = source or root / "backends" / "cpp" / "topoml_native.cpp"
    if not src.exists():
        raise FileNotFoundError(src)
    selected = compiler or _find_cxx_compiler()
    if selected is None:
        raise NativeBackendUnavailable("no C++ compiler found; expected c++, g++, or clang++")
    out_dir = build_dir or root / "artifacts" / "native"
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / _library_name("topoml_native")
    command = (
        selected,
        "-std=c++17",
        "-O2",
        "-fPIC",
        "-shared",
        str(src),
        "-o",
        str(library),
    )
    env = os.environ.copy()
    subprocess.run(command, check=True, cwd=root, env=env)
    return NativeBuildResult(source=src, library=library, compiler=selected, command=command)


def load_cpp_native_backend(library: Path) -> CppNativeBackend:
    return CppNativeBackend(library)


def _find_cxx_compiler() -> str | None:
    for candidate in ("c++", "g++", "clang++"):
        found = shutil.which(candidate)
        if found is not None:
            return found
    return None


def _library_name(stem: str) -> str:
    system = platform.system().lower()
    if system == "windows":
        return f"{stem}.dll"
    if system == "darwin":
        return f"lib{stem}.dylib"
    return f"lib{stem}.so"


def _as_points(points: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype=dtype)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] == 0:
        raise ValueError("points must be a non-empty 2D array")
    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")
    return pts
