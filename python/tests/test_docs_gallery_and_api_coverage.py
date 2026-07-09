from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read_doc(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_gallery_documents_many_graph_forms_and_claim_boundaries() -> None:
    pages = [
        "docs/gallery/noisy-circles.md",
        "docs/gallery/time-delay.md",
        "docs/gallery/mapper-reeb-activation-map.md",
        "docs/gallery/sheaf-consistency-residuals.md",
        "docs/gallery/covers-nerves-routing.md",
        "docs/gallery/persistence-barcode-betti-curves.md",
        "docs/gallery/manifold-embedding-neighborhood-graph.md",
        "docs/gallery/backend-runtime-gates.md",
        "docs/gallery/topology-training-pipeline.md",
        "docs/gallery/tda-benchmark-evidence-map.md",
        "docs/gallery/finite-topology-dynamics-and-braids.md",
    ]
    gallery_text = ""
    for page in pages:
        text = _read_doc(page)
        assert "```mermaid" in text, page
        assert "Claim Boundary" in text or "Claim boundary" in text, page
        gallery_text += "\n" + text

    for graph_kind in [
        "flowchart",
        "graph",
        "sequenceDiagram",
        "stateDiagram-v2",
        "classDiagram",
        "timeline",
        "quadrantChart",
        "journey",
    ]:
        assert graph_kind in gallery_text


def test_api_docs_cover_public_surface_added_for_backends_and_training() -> None:
    api = _read_doc("docs/reference/api.md")
    required_symbols = [
        "persistent_homology",
        "time_delay_embedding",
        "PHFeaturizer",
        "BettiCurve",
        "PersistenceImage",
        "BenchmarkDataset",
        "list_benchmark_datasets",
        "load_benchmark_dataset",
        "make_noisy_circle",
        "make_two_circles",
        "make_cluster_bridge",
        "metric_cover",
        "nerve_graph",
        "mapper_graph",
        "sheaf_consistency_residual",
        "path_homotopy_signature",
        "activation_strata",
        "finite_orbit_signature",
        "equivariance_residual",
        "scott_fixed_point",
        "weak_convergence_residual",
        "finite_topology_signature",
        "dynamical_signature",
        "braid_crossing_signature",
        "mesh_euler_characteristic",
        "TensorBundleSpec",
        "TensorAlgebraElement",
        "interop_bundle",
        "interop_add",
        "TopologyAugmenter",
        "TopologyRandomForestClassifier",
        "topological_sample_weights",
        "make_sklearn_pipeline",
        "SklearnUnavailableError",
        "build_cpp_native_backend",
        "load_cpp_native_backend",
        "CppNativeBackend",
        "build_asm_native_backend",
        "load_asm_native_backend",
        "AsmNativeBackend",
        "build_cuda_native_backend",
        "load_cuda_native_backend",
        "CudaNativeBackend",
        "triton_runtime_status",
        "triton_pairwise_l2",
        "TorchTensorAdapter",
        "TensorFlowTensorAdapter",
    ]
    for symbol in required_symbols:
        assert symbol in api


def test_backend_compatibility_matrix_is_explicit() -> None:
    text = _read_doc("docs/backends/compatibility.md")
    for backend in [
        "Safe Rust",
        "Python reference",
        "C++",
        "ASM AVX-512",
        "CUDA",
        "Triton",
        "PyTorch",
        "TensorFlow",
    ]:
        assert backend in text
    for phrase in ["runtime gate", "claim boundary", "fallback", "verification"]:
        assert phrase in text.lower()


def test_benchmark_dataset_docs_are_explicit() -> None:
    text = _read_doc("docs/benchmarks/datasets.md")
    for phrase in [
        "make_noisy_circle",
        "make_two_circles",
        "make_cluster_bridge",
        "expected_betti",
        "Claim Boundary",
    ]:
        assert phrase in text
