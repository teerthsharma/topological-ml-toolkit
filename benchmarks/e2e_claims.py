from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

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


def _claim_feature_encoders_and_signatures() -> dict:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)
    curve = topoml.BettiCurve(radii=[0.0, 0.3, 6.0], homology_dims=[0])
    curve_features = curve.fit_transform([diagram])
    image = topoml.PersistenceImage(width=4, height=3, sigma=0.2).fit_transform([diagram])
    point_sig = topoml.point_cloud_signature(points, radii=[0.0, 0.3], max_dim=0)
    graph_sig = topoml.graph_signature(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float))
    activation_sig = topoml.activation_signature(
        np.array([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]], dtype=float),
        radii=[0.0, 1.1],
        max_dim=0,
    )

    assert curve_features.tolist() == [[3.0, 2.0, 1.0]]
    assert image.shape == (1, 12)
    assert image.sum() > 0.0
    assert point_sig.values["beta0@0.3"] == 2.0
    assert graph_sig.values["cycle_rank"] == 1.0
    assert activation_sig.values["samples"] == 4.0
    return {
        "betti_curve": curve_features.tolist(),
        "persistence_image_shape": list(image.shape),
        "point_cloud_signature": point_sig.values,
        "graph_signature": graph_sig.values,
        "activation_signature": activation_sig.values,
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


def _claim_backend_source_inventory() -> dict:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "backends" / "cuda" / "topology_distance.cu",
        root / "backends" / "cuda" / "warp_reductions.cu",
        root / "backends" / "asm" / "x86_64_l2_f32.S",
        root / "backends" / "asm" / "x86_64_dispatch.S",
        root / "backends" / "cpp" / "topoml_native.cpp",
        root / "backends" / "triton" / "topology_distance.py",
        root / "benchmarks" / "benchmark_native_distance.py",
        root / "benchmarks" / "benchmark_asm_distance.py",
        root / "benchmarks" / "benchmark_ml_adapters.py",
        root / "python" / "topoml" / "native.py",
        root / "python" / "topoml" / "asm.py",
        root / "python" / "tests" / "test_cpp_native_ctypes.py",
        root / "python" / "tests" / "test_asm_native_ctypes.py",
        root / "python" / "tests" / "test_framework_adapters.py",
        root / "python" / "tests" / "test_gpu_backend_semantic_contract.py",
    ]
    sizes = {}
    for path in files:
        assert path.exists(), path
        size = path.stat().st_size
        assert size > 0, path
        sizes[str(path.relative_to(root)).replace("\\", "/")] = size
    return {
        "source_files": sizes,
        "claim_scope": "backend source plus native C++, Linux x86-64 ASM, real CPU ML adapter integration, and CPU GPU-kernel semantic fixture coverage; runtime selection remains gated",
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


def _claim_topology_prototypes() -> dict:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0]], dtype=float)
    cover = topoml.metric_cover(points, radius=0.25)
    nerve = topoml.nerve_graph(cover)
    mapper = topoml.mapper_graph(
        np.array([[0.0], [0.4], [0.8]], dtype=float),
        np.array([0.0, 0.4, 0.8], dtype=float),
        intervals=2,
        overlap=0.75,
        cluster_radius=1.0,
    )
    residual = topoml.sheaf_consistency_residual(
        {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 3.0])},
        [("a", "b", np.eye(2))],
    )

    assert nerve.edges == ((0, 1),)
    assert mapper.edges == ((0, 1),)
    assert residual.max_residual == 1.0
    return {
        "cover_cells": [list(cell.members) for cell in cover.cells],
        "nerve_edges": [list(edge) for edge in nerve.edges],
        "mapper_edges": [list(edge) for edge in mapper.edges],
        "sheaf_max_residual": residual.max_residual,
        "claim_scope": "prototype topology diagnostics, not accelerated backend behavior",
    }


def _claim_tensor_bundle_and_training_surface() -> dict:
    xy = topoml.TensorBundleSpec(("x", "y"), (1.0, 1.0))
    yz = topoml.TensorBundleSpec(("y", "z"), (1.0, -1.0), tangent_order=1, tangent_variables=2)
    ambient = topoml.interop_bundle(xy, yz)
    summed = topoml.interop_add(
        topoml.TensorAlgebraElement(xy, np.array([1.0, 2.0])),
        topoml.TensorAlgebraElement(yz, np.array([3.0, 4.0])),
    )

    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
        np.array([[1.0, 1.0], [1.1, 1.0], [8.0, 1.0]], dtype=float),
        np.array([[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]], dtype=float),
    ]
    labels = np.array(["separated", "connected", "separated", "connected"], dtype=object)
    augmenter = topoml.TopologyAugmenter(radii=[0.0, 0.15, 1.0], max_dim=0)
    augmented = augmenter.fit_transform(clouds)
    weights = topoml.topological_sample_weights(clouds, radii=[0.0, 0.15, 1.0], max_dim=0)
    model = topoml.TopologyRandomForestClassifier(
        n_estimators=15,
        max_features=2,
        radii=[0.0, 0.15, 1.0],
        max_dim=0,
        random_state=7,
    )
    model.fit(clouds, labels, sample_weight=weights)
    score = model.score(clouds, labels)

    assert ambient.basis == ("x", "y", "z")
    assert summed.coordinates.tolist() == [1.0, 5.0, 4.0]
    assert augmented.shape == (4, 3)
    assert np.isclose(weights.mean(), 1.0)
    assert score == 1.0
    return {
        "ambient_basis": list(ambient.basis),
        "ambient_metric": list(ambient.metric),
        "summed_coordinates": summed.coordinates.tolist(),
        "augmented_shape": list(augmented.shape),
        "sample_weights": weights.tolist(),
        "training_score": score,
        "claim_scope": "dependency-light topology training baseline, not a deep-learning speedup claim",
    }


def _claim_topology_family_registry() -> dict:
    families = {family.id: family for family in topoml.topology_families()}
    required = {
        "point_set",
        "metric_uniform_proximity",
        "domain_scott",
        "homotopy",
        "cohomology",
        "sheaves_cosheaves",
        "mapper_reeb_contour",
        "morse_conley_dynamical",
        "stratified_singular",
        "nerve_cech_cover",
        "knot_braid_link",
        "fiber_bundles",
        "groups_actions_quotients",
        "low_dimensional_geometric",
        "categorical_pointless",
        "topological_vector_function_space",
    }
    assert set(families) == required
    assert families["mapper_reeb_contour"].status == "prototype"
    assert "mapper_graph" in families["mapper_reeb_contour"].api_surface
    matrix = topoml.topology_family_coverage_matrix()
    assert len(matrix) == len(required)
    return {
        "families": sorted(families),
        "prototype_families": sorted(family.id for family in families.values() if family.status == "prototype"),
        "coverage_rows": len(matrix),
        "claim_scope": "coverage registry and docs/prototype status, not full mathematical implementation",
    }


def _claim_framework_adapter_import_safety() -> dict:
    from topoml.adapters import (
        TensorFlowActivationCapture,
        TensorFlowTensorAdapter,
        TorchActivationCapture,
        TorchTensorAdapter,
    )

    forbidden = {"torch", "tensorflow"}
    loaded = sorted(forbidden.intersection(sys.modules))
    assert loaded == []
    return {
        "adapter_classes": [
            TorchTensorAdapter.__name__,
            TorchActivationCapture.__name__,
            TensorFlowTensorAdapter.__name__,
            TensorFlowActivationCapture.__name__,
        ],
        "heavy_modules_loaded": loaded,
        "claim_scope": "optional adapter API import safety; runtime parity is tested when dependencies exist",
    }


def _claim_visual_topology_gallery_docs() -> dict:
    root = Path(__file__).resolve().parents[1]
    pages = {
        "mapper": root / "docs" / "gallery" / "mapper-reeb-activation-map.md",
        "sheaf": root / "docs" / "gallery" / "sheaf-consistency-residuals.md",
        "cover": root / "docs" / "gallery" / "covers-nerves-routing.md",
    }
    required_phrases = {
        "mapper": ("Mapper graph", "Claim Boundary", "topoml.mapper_graph"),
        "sheaf": ("sheaf_consistency_residual", "Claim Boundary", "Residual"),
        "cover": ("metric_cover", "Benchmark Gate", "same-budget random"),
    }
    evidence = {}
    for key, path in pages.items():
        assert path.exists(), path
        text = path.read_text(encoding="utf-8")
        for phrase in required_phrases[key]:
            assert phrase in text, f"{phrase!r} missing from {path}"
        evidence[key] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": len(text.encode("utf-8")),
            "has_mermaid": "```mermaid" in text,
        }
    return {
        "pages": evidence,
        "claim_scope": "visual docs for active prototypes with explicit claim boundaries",
    }


def _claim_dashboard_export() -> dict:
    points = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=2.0)
    features = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.5]).fit_transform([points])
    with TemporaryDirectory() as tmp:
        path = topoml.write_dashboard(
            Path(tmp) / "dashboard.html",
            title="E2E topology dashboard",
            diagram=diagram,
            feature_matrix=features,
            metadata={"source": "e2e_claims"},
        )
        html = path.read_text(encoding="utf-8")
    assert "E2E topology dashboard" in html
    assert "Persistence Diagram" in html
    assert "Feature Matrix" in html
    return {
        "html_bytes": len(html.encode("utf-8")),
        "self_contained": "plotly" not in html.lower(),
        "claim_scope": "static GUI export for inspection, not a hosted web application",
    }


def run_claims() -> list[ClaimResult]:
    return [
        _record("H0 cluster merges match known Betti numbers", _claim_h0_cluster_merges),
        _record("H1 square cycle appears and is filled by diagonals/triangles", _claim_h1_square_cycle),
        _record("Time-delay embedding returns graph-ready delay vectors", _claim_time_delay_embedding),
        _record("PHFeaturizer exports fixed-width ML feature matrices", _claim_ph_featurizer),
        _record(
            "Feature encoders and topology signatures cover diagrams, graphs, and activations",
            _claim_feature_encoders_and_signatures,
        ),
        _record("Topology prototype APIs build covers, Mapper edges, and sheaf residuals", _claim_topology_prototypes),
        _record(
            "TensorBundle interoperability and topology-aware training baseline execute",
            _claim_tensor_bundle_and_training_surface,
        ),
        _record("Topology family coverage registry includes the objective taxonomy", _claim_topology_family_registry),
        _record("PyTorch and TensorFlow adapter APIs import without loading heavy stacks", _claim_framework_adapter_import_safety),
        _record("Visual topology gallery documents active Mapper, sheaf, and cover prototypes", _claim_visual_topology_gallery_docs),
        _record("GUI exporter writes a self-contained topology dashboard", _claim_dashboard_export),
        _record("Backend metadata separates active code from planned acceleration", _claim_backend_contract),
        _record("Planned backend source files exist for CUDA, ASM, C++, and Triton", _claim_backend_source_inventory),
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
