from __future__ import annotations

import ctypes
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .native import NativeBackendUnavailable, _library_name


@dataclass(frozen=True)
class AsmBuildResult:
    sources: tuple[Path, ...]
    library: Path
    compiler: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class CpuFeatureProbe:
    leaf7_ebx: int
    avx2: bool
    avx512f: bool


class AsmNativeBackend:
    """ctypes wrapper around the x86-64 assembly dispatch probes."""

    def __init__(self, library: Path):
        self.library = Path(library)
        self._lib = ctypes.CDLL(str(self.library))
        self._lib.topoml_cpuid_leaf7_ebx.argtypes = []
        self._lib.topoml_cpuid_leaf7_ebx.restype = ctypes.c_uint32
        self._lib.topoml_has_avx2.argtypes = []
        self._lib.topoml_has_avx2.restype = ctypes.c_uint32
        self._lib.topoml_has_avx512f.argtypes = []
        self._lib.topoml_has_avx512f.restype = ctypes.c_uint32
        self._lib.topoml_l2_sq_f32_asm.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_long,
        ]
        self._lib.topoml_l2_sq_f32_asm.restype = ctypes.c_float

    def cpu_features(self) -> CpuFeatureProbe:
        leaf7 = int(self._lib.topoml_cpuid_leaf7_ebx())
        avx2 = bool(self._lib.topoml_has_avx2())
        avx512f = bool(self._lib.topoml_has_avx512f())
        return CpuFeatureProbe(leaf7_ebx=leaf7, avx2=avx2, avx512f=avx512f)

    def l2_sq_f32(self, left: np.ndarray, right: np.ndarray) -> float:
        lhs = _as_vector(left)
        rhs = _as_vector(right)
        if lhs.shape != rhs.shape:
            raise ValueError("left and right vectors must have the same shape")
        return float(
            self._lib.topoml_l2_sq_f32_asm(
                lhs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                rhs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_long(lhs.size),
            )
        )


def build_asm_native_backend(
    *,
    sources: tuple[Path, ...] | None = None,
    build_dir: Path | None = None,
    compiler: str | None = None,
) -> AsmBuildResult:
    if platform.system().lower() != "linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise NativeBackendUnavailable("assembly smoke gate currently supports Linux x86-64 only")
    root = Path(__file__).resolve().parents[2]
    srcs = sources or (
        root / "backends" / "asm" / "x86_64_dispatch.S",
        root / "backends" / "asm" / "x86_64_l2_f32.S",
    )
    for src in srcs:
        if not src.exists():
            raise FileNotFoundError(src)
    selected = compiler or _find_c_compiler()
    if selected is None:
        raise NativeBackendUnavailable("no C compiler found; expected cc, gcc, or clang")
    out_dir = build_dir or root / "artifacts" / "native"
    out_dir.mkdir(parents=True, exist_ok=True)
    library = out_dir / _library_name("topoml_asm")
    command = (
        selected,
        "-O2",
        "-fPIC",
        "-shared",
        *(str(src) for src in srcs),
        "-o",
        str(library),
    )
    subprocess.run(command, check=True, cwd=root)
    return AsmBuildResult(sources=srcs, library=library, compiler=selected, command=command)


def load_asm_native_backend(library: Path) -> AsmNativeBackend:
    return AsmNativeBackend(library)


def _find_c_compiler() -> str | None:
    for candidate in ("cc", "gcc", "clang"):
        found = shutil.which(candidate)
        if found is not None:
            return found
    return None


def _as_vector(values: np.ndarray) -> np.ndarray:
    vector = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        raise ValueError("vector must be non-empty")
    if not np.isfinite(vector).all():
        raise ValueError("vector must contain only finite values")
    return vector
