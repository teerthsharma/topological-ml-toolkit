from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyFamily:
    id: str
    name: str
    object_or_invariant: str
    ml_or_systems_use: str
    status: str
    docs_path: str
    api_surface: tuple[str, ...] = ()
    benchmark_status: str = "not benchmarked"


_FAMILIES: tuple[TopologyFamily, ...] = (
    TopologyFamily(
        "point_set",
        "Point-set / general topology",
        "Open sets, neighborhoods, continuity, compactness, separation, connectedness",
        "Data validity, convergence contracts, finite covers, and preprocessing continuity.",
        "docs",
        "docs/topology/foundations.md",
    ),
    TopologyFamily(
        "metric_uniform_proximity",
        "Metric, uniform, and proximity spaces",
        "Entourages, Cauchy behavior, total boundedness, nearness without one fixed metric",
        "Embedding tolerance, approximate nearest-neighbor behavior, drift, and cache locality.",
        "prototype",
        "docs/topology/metric-and-function-spaces.md",
        ("metric_cover", "nerve_graph"),
        "E2E metric-cover and nerve checks",
    ),
    TopologyFamily(
        "domain_scott",
        "Domain theory / Scott topology",
        "Posets, directed suprema, monotone maps, fixed points, computable continuity",
        "Compiler semantics, schedulers, streaming dataflow, and monotone runtime proofs.",
        "docs",
        "docs/topology/systems-topology.md",
    ),
    TopologyFamily(
        "homotopy",
        "Homotopy theory",
        "Fundamental group, higher homotopy groups, path classes, deformation classes",
        "Optimization and system-state trajectories where path class matters.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
    ),
    TopologyFamily(
        "cohomology",
        "Cohomology beyond Betti counts",
        "Cocycles, cup products, obstruction classes",
        "Feature interactions and distributed consistency beyond scalar Betti counts.",
        "prototype",
        "docs/topology/systems-topology.md",
        ("sheaf_consistency_residual",),
        "E2E sheaf residual check",
    ),
    TopologyFamily(
        "sheaves_cosheaves",
        "Sheaves and cosheaves",
        "Local data glued by restriction maps; sheaf cohomology residuals",
        "Layers, heads, shards, feature stores, sensors, and worker consistency.",
        "prototype",
        "docs/topology/systems-topology.md",
        ("sheaf_consistency_residual",),
        "E2E sheaf residual check",
    ),
    TopologyFamily(
        "mapper_reeb_contour",
        "Mapper / Reeb / contour topology",
        "Level-set graphs of scalar filters; critical changes in shape",
        "Interpretable maps of datasets, activations, loss surfaces, and routing states.",
        "prototype",
        "docs/topology/systems-topology.md",
        ("mapper_graph",),
        "E2E Mapper graph check",
    ),
    TopologyFamily(
        "morse_conley_dynamical",
        "Morse / Conley / dynamical topology",
        "Critical points, gradient flows, invariant sets, attractor decompositions",
        "Training dynamics, online learning regimes, and optimizer stability.",
        "docs",
        "docs/topology/systems-topology.md",
    ),
    TopologyFamily(
        "stratified_singular",
        "Stratified and singular spaces",
        "Spaces with strata, boundaries, corners, singularities",
        "ReLU regions, decision boundaries, failure surfaces, and non-manifold data.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
    ),
    TopologyFamily(
        "nerve_cech_cover",
        "Nerve / Cech / cover calculus",
        "Covers and nerves; topology-preserving summaries",
        "Cover-based batching, cache regions, privacy summaries, and routing cells.",
        "prototype",
        "docs/topology/systems-topology.md",
        ("metric_cover", "nerve_graph"),
        "E2E metric-cover and nerve checks",
    ),
    TopologyFamily(
        "knot_braid_link",
        "Knot, braid, and link topology",
        "Embedded curves, crossings, linking and braid invariants",
        "Thread interleavings, trajectory entanglement, and attention-head coupling.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
    ),
    TopologyFamily(
        "fiber_bundles",
        "Fiber bundles and characteristic classes",
        "Fibers over base spaces, sections, holonomy, characteristic obstructions",
        "Gauge/equivariant ML, layerwise parameter transport, and symmetry defects.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
        ("TensorBundleSpec", "interop_bundle", "interop_add"),
        "E2E TensorBundle interop check",
    ),
    TopologyFamily(
        "groups_actions_quotients",
        "Topological groups, group actions, quotients",
        "Orbits, stabilizers, quotient spaces",
        "Symmetry-aware model tests, equivariance contracts, and quotient embeddings.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
    ),
    TopologyFamily(
        "low_dimensional_geometric",
        "Low-dimensional / geometric topology",
        "Orientability, genus, handles, surgery, 3/4-manifold decompositions",
        "Mesh systems, memory-layout surgery, graph rewrites, and scene diagnostics.",
        "docs",
        "docs/topology/geometry-and-symmetry.md",
    ),
    TopologyFamily(
        "categorical_pointless",
        "Categorical, Grothendieck, and pointless topology",
        "Sites, locales, topoi, open-set lattices instead of point sets",
        "Compositional observability, privacy, distributed knowledge, and sheaf APIs.",
        "docs",
        "docs/topology/foundations.md",
    ),
    TopologyFamily(
        "topological_vector_function_space",
        "Topological vector spaces / function-space topology",
        "Weak/strong topologies, convergence of functions and distributions",
        "Model convergence, kernel methods, infinite-width limits, and distribution shift.",
        "docs",
        "docs/topology/metric-and-function-spaces.md",
    ),
)


def topology_families() -> tuple[TopologyFamily, ...]:
    return _FAMILIES


def topology_family(family_id: str) -> TopologyFamily:
    normalized = family_id.strip().lower().replace("-", "_")
    for family in _FAMILIES:
        if family.id == normalized:
            return family
    raise KeyError(f"Unknown topology family: {family_id!r}")


def topology_family_coverage_matrix() -> list[dict[str, object]]:
    return [
        {
            "id": family.id,
            "name": family.name,
            "status": family.status,
            "docs_path": family.docs_path,
            "api_surface": list(family.api_surface),
            "benchmark_status": family.benchmark_status,
        }
        for family in _FAMILIES
    ]
