from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native import NativeBackendUnavailable, _as_points, _library_name


@dataclass(frozen=True)
class CudaBuildResult:
    source: Path
    library: Path
    compiler: str
    command: tuple[str, ...]


class CudaNativeBackend:
    """ctypes wrapper around the optional CUDA host launcher ABI."""

    def __init__(self, library: Path):
        self.library = Path(library)
        try:
            self._lib = ctypes.CDLL(str(self.library))
        except OSError as exc:
            raise NativeBackendUnavailable(f"could not load CUDA native library {self.library}: {exc}") from exc
        self._lib.topoml_cuda_pairwise_l2_f32_host.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib.topoml_cuda_pairwise_l2_f32_host.restype = ctypes.c_int
        self._lib.topoml_cuda_threshold_edges_u8_host.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int64,
            ctypes.c_float,
        ]
        self._lib.topoml_cuda_threshold_edges_u8_host.restype = ctypes.c_int

    def pairwise_l2(self, points: np.ndarray) -> np.ndarray:
        pts = _as_points(points, dtype=np.float32)
        out = np.empty((pts.shape[0], pts.shape[0]), dtype=np.float32)
        code = self._lib.topoml_cuda_pairwise_l2_f32_host(
            pts.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int64(pts.shape[0]),
            ctypes.c_int64(pts.shape[1]),
        )
        if code != 0:
            raise NativeBackendUnavailable(f"CUDA pairwise L2 returned cudaError code {code}")
        return out

    def threshold_edges(self, distances: np.ndarray, radius: float) -> np.ndarray:
        dists = np.ascontiguousarray(distances, dtype=np.float32)
        if dists.ndim != 2 or dists.shape[0] == 0 or dists.shape[0] != dists.shape[1]:
            raise ValueError("distances must be a non-empty square matrix")
        if radius < 0.0:
            raise ValueError("radius must be non-negative")
        edges = np.empty(dists.shape, dtype=np.uint8)
        code = self._lib.topoml_cuda_threshold_edges_u8_host(
            dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            edges.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_int64(dists.shape[0]),
            ctypes.c_float(float(radius)),
        )
        if code != 0:
            raise NativeBackendUnavailable(f"CUDA threshold edges returned cudaError code {code}")
        return edges


def build_cuda_native_backend(
    *,
    source: Path | None = None,
    build_dir: Path | None = None,
    compiler: str | None = None,
) -> CudaBuildResult:
    root = Path(__file__).resolve().parents[2]
    src = source or root / "backends" / "cuda" / "topology_distance.cu"
    if not src.exists():
        raise FileNotFoundError(src)
    selected = compiler or shutil.which("nvcc")
    if selected is None:
        raise NativeBackendUnavailable("nvcc is not available")
    if sys.platform == "win32" and shutil.which("cl.exe") is None:
        raise NativeBackendUnavailable("nvcc is available, but the MSVC host compiler cl.exe is not on PATH")
    out_dir = build_dir or root / "artifacts" / "native"
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / _library_name("topoml_cuda")
    command = (
        selected,
        "-std=c++17",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        str(src),
        "-o",
        str(library),
    )
    subprocess.run(command, check=True, cwd=root)
    return CudaBuildResult(source=src, library=library, compiler=selected, command=command)


def load_cuda_native_backend(library: Path) -> CudaNativeBackend:
    return CudaNativeBackend(library)
