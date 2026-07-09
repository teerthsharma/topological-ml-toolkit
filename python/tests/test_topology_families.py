import topoml


OBJECTIVE_FAMILY_IDS = {
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


def test_topology_family_registry_covers_objective_families():
    families = {family.id: family for family in topoml.topology_families()}

    assert OBJECTIVE_FAMILY_IDS == set(families)
    assert families["mapper_reeb_contour"].status == "prototype"
    assert "mapper_graph" in families["mapper_reeb_contour"].api_surface
    assert families["homotopy"].status == "prototype"
    assert "path_homotopy_signature" in families["homotopy"].api_surface
    assert families["domain_scott"].status == "prototype"
    assert "scott_fixed_point" in families["domain_scott"].api_surface
    assert families["stratified_singular"].status == "prototype"
    assert "activation_strata" in families["stratified_singular"].api_surface
    assert families["groups_actions_quotients"].status == "prototype"
    assert "finite_orbit_signature" in families["groups_actions_quotients"].api_surface
    assert "equivariance_residual" in families["groups_actions_quotients"].api_surface
    assert families["topological_vector_function_space"].status == "prototype"
    assert "weak_convergence_residual" in families["topological_vector_function_space"].api_surface
    assert families["fiber_bundles"].status == "docs"
    assert "TensorBundleSpec" in families["fiber_bundles"].api_surface
    assert families["point_set"].status == "docs"


def test_topology_family_coverage_matrix_is_public_and_auditable():
    rows = topoml.topology_family_coverage_matrix()

    assert len(rows) == len(OBJECTIVE_FAMILY_IDS)
    assert all({"id", "name", "status", "docs_path", "api_surface", "benchmark_status"}.issubset(row) for row in rows)
    assert topoml.topology_family("nerve-cech-cover").id == "nerve_cech_cover"
