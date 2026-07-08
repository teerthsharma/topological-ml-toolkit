from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import topoml


@dataclass(frozen=True)
class ClaimResult:
    claim: str
    status: str
    evidence: dict


def _record(claim: str, fn) -> ClaimResult:
    start = time.perf_counter()
    try:
        evidence = fn()
        evidence["seconds"] = round(time.perf_counter() - start, 9)
        return ClaimResult(claim=claim, status="pass", evidence=evidence)
    except Exception as exc:  # pragma: no cover - exercised by CI failure path
        return ClaimResult(
            claim=claim,
            status="fail",
            evidence={"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - start, 9)},
        )


def _claim_h0_cluster_merges() -> dict:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)
    observed = {
        "r=0.1": diagram.betti_at(0.1).beta0,
        "r=0.3": diagram.betti_at(0.3).beta0,
        "r=6.0": diagram.betti_at(6.0).beta0,
    }
    expected = {"r=0.1": 3, "r=0.3": 2, "r=6.0": 1}
    assert observed == expected
    return {"expected": expected, "observed": observed, "pairs": len(diagram.pairs)}


def _claim_h1_square_cycle() -> dict:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    diagram = topoml.persistent_homology(square, max_dim=1, max_radius=2.0)
    observed = {"r=1.1": diagram.betti_at(1.1).beta1, "r=1.5": diagram.betti_at(1.5).beta1}
    expected = {"r=1.1": 1, "r=1.5": 0}
    assert observed == expected
    return {"expected": expected, "observed": observed, "pairs": len(diagram.pairs)}


def _claim_time_delay_embedding() -> dict:
    samples = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0], dtype=float)
    cloud = topoml.time_delay_embedding(samples, dimension=3, tau=1)
    assert cloud.shape == (4, 3)
    assert cloud[0].tolist() == [0.0, 1.0, 0.0]
    return {"shape": list(cloud.shape), "first_vector": cloud[0].tolist()}


def _claim_ph_featurizer() -> dict:
    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
    ]
    featurizer = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.15, 1.0])
    features = featurizer.fit_transform(clouds)
    assert features.shape == (2, 3)
    assert features.tolist() == [[3.0, 2.0, 2.0], [3.0, 3.0, 1.0]]
    return {
        "shape": list(features.shape),
        "feature_names": featurizer.get_feature_names_out().tolist(),
        "matrix": features.tolist(),
    }


def _claim_backend_contract() -> dict:
    metadata = {backend.id: backend for backend in topoml.available_backends()}
    active = {backend.id for backend in metadata.values() if backend.active and backend.available}
    planned = {backend.id for backend in metadata.values() if backend.planned and not backend.available}
    assert {"safe_rust", "python_reference"}.issubset(active)
    assert {"cpp", "asm_avx512", "triton", "pytorch", "tensorflow"}.issubset(planned)
    assert topoml.select_backend("triton") is None
    adapter_result = topoml.select_backend_adapter("triton", raise_unavailable=False)
    assert adapter_result.adapter.id == "triton"
    assert not adapter_result.available
    assert adapter_result.missing_gates
    return {
        "active": sorted(active),
        "planned_unavailable": sorted(planned),
        "strict_adapter_example": {
            "id": adapter_result.adapter.id,
            "available": adapter_result.available,
            "missing_gates": list(adapter_result.missing_gates),
        },
    }


def _claim_import_guard() -> dict:
    forbidden = {"torch", "tensorflow", "triton"}
    loaded = sorted(forbidden.intersection(sys.modules))
    assert loaded == []
    return {"forbidden_optional_modules_loaded": loaded}


def _claim_benchmark_record() -> dict:
    rng = np.random.default_rng(42)
    rows = []
    for n in (8, 12):
        points = rng.normal(size=(n, 2))
        start = time.perf_counter()
        diagram = topoml.persistent_homology(points, max_dim=1, max_radius=1.0)
        rows.append(
            {
                "backend": "python_reference",
                "points": n,
                "seconds": round(time.perf_counter() - start, 9),
                "pairs": len(diagram.pairs),
            }
        )
    assert all(row["seconds"] >= 0.0 and row["pairs"] > 0 for row in rows)
    return {"rows": rows, "claim_scope": "timed smoke record, not a speedup claim"}


def run_claims() -> list[ClaimResult]:
    return [
        _record("H0 cluster merges match known Betti numbers", _claim_h0_cluster_merges),
        _record("H1 square cycle appears and is filled by diagonals/triangles", _claim_h1_square_cycle),
        _record("Time-delay embedding returns graph-ready delay vectors", _claim_time_delay_embedding),
        _record("PHFeaturizer exports fixed-width ML feature matrices", _claim_ph_featurizer),
        _record("Backend metadata separates active code from planned acceleration", _claim_backend_contract),
        _record("Importing topoml does not import heavy optional ML/GPU stacks", _claim_import_guard),
        _record("Benchmark runner records active Python-reference timings", _claim_benchmark_record),
    ]


def write_markdown(results: list[ClaimResult], path: Path) -> None:
    lines = [
        "# E2E Claim Benchmark",
        "",
        "This report is generated by `benchmarks/e2e_claims.py`. It verifies active repository claims and records benchmark-smoke timings without turning them into speedup claims.",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "| Claim | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for result in results:
        evidence = json.dumps(result.evidence, sort_keys=True)
        lines.append(f"| {result.claim} | {result.status} | `{evidence}` |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/e2e-claims.json"))
    parser.add_argument("--md-out", type=Path, default=Path("artifacts/e2e-claims.md"))
    args = parser.parse_args()

    results = run_claims()
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "results": [asdict(result) for result in results],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(results, args.md_out)
    print(json.dumps(payload, indent=2))

    if any(result.status != "pass" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
