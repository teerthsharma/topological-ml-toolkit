from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

import topoml


def run(points: list[int], dims: int, out: Path) -> dict:
    build = topoml.build_asm_native_backend(build_dir=Path("artifacts/native"))
    backend = topoml.load_asm_native_backend(build.library)
    features = backend.cpu_features()
    rng = np.random.default_rng(20260709)
    rows = []
    for n in points:
        cloud = rng.normal(size=(n, dims)).astype(np.float32)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        start = time.perf_counter()
        asm_results = [backend.l2_sq_f32_dispatched(cloud[i], cloud[j]) for i, j in pairs]
        asm_seconds = time.perf_counter() - start
        asm_values = [result.value for result in asm_results]
        dispatch_backends = sorted({result.backend for result in asm_results})
        start = time.perf_counter()
        numpy_values = [float(np.sum((cloud[i] - cloud[j]) ** 2, dtype=np.float32)) for i, j in pairs]
        numpy_seconds = time.perf_counter() - start
        if not np.allclose(np.array(asm_values), np.array(numpy_values), rtol=1e-6, atol=1e-6):
            raise AssertionError(f"ASM L2 mismatch for n={n}, dims={dims}")
        rows.append(
            {
                "backend": "asm-x86_64-dispatched",
                "points": n,
                "dims": dims,
                "pairs": len(pairs),
                "dispatch_backends": dispatch_backends,
                "used_avx512": any(result.used_avx512 for result in asm_results),
                "asm_seconds": round(asm_seconds, 9),
                "numpy_seconds": round(numpy_seconds, 9),
            }
        )
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "library": str(build.library),
        "compiler": build.compiler,
        "cpu_features": {
            "leaf7_ebx": features.leaf7_ebx,
            "xcr0": features.xcr0,
            "avx2": features.avx2,
            "avx512f": features.avx512f,
            "avx512f_os": features.avx512f_os,
        },
        "rows": rows,
        "claim_scope": "ASM CPUID/OS-gated scalar-or-AVX-512 L2 correctness smoke; not a persistent-homology or speedup claim",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/asm-distance.json"))
    parser.add_argument("--points", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--dims", type=int, default=8)
    args = parser.parse_args()
    run(args.points, args.dims, args.out)


if __name__ == "__main__":
    main()
