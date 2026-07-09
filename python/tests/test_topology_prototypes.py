import numpy as np

import topoml


def test_metric_cover_builds_cells_and_nerve_edges():
    points = np.array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0]], dtype=float)

    cover = topoml.metric_cover(points, radius=0.25)
    nerve = topoml.nerve_graph(cover)

    assert cover.cells[0].members == (0, 1)
    assert cover.cells[2].members == (2,)
    assert nerve.nodes == (0, 1, 2)
    assert nerve.edges == ((0, 1),)


def test_mapper_graph_connects_overlapping_filter_components():
    points = np.array([[0.0], [0.1], [1.0], [1.1]], dtype=float)
    filters = np.array([0.0, 0.1, 1.0, 1.1], dtype=float)

    graph = topoml.mapper_graph(points, filters, intervals=2, overlap=0.25, cluster_radius=0.2)

    assert len(graph.nodes) == 2
    assert graph.nodes[0].members == (0, 1)
    assert graph.nodes[1].members == (2, 3)
    assert graph.edges == ()


def test_mapper_graph_records_overlap_edges():
    points = np.array([[0.0], [0.4], [0.8]], dtype=float)
    filters = np.array([0.0, 0.4, 0.8], dtype=float)

    graph = topoml.mapper_graph(points, filters, intervals=2, overlap=0.75, cluster_radius=1.0)

    assert len(graph.nodes) == 2
    assert graph.edges == ((0, 1),)


def test_sheaf_consistency_residual_flags_disagreement():
    residual = topoml.sheaf_consistency_residual(
        {
            "layer0": np.array([1.0, 2.0]),
            "layer1": np.array([1.0, 3.0]),
        },
        [("layer0", "layer1", np.eye(2))],
    )

    assert residual.edge_residuals == {"layer0->layer1": 1.0}
    assert residual.max_residual == 1.0
    assert residual.mean_residual == 1.0


def test_path_homotopy_signature_counts_planar_winding():
    path = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )

    signature = topoml.path_homotopy_signature(path)

    assert signature.closed
    assert signature.winding_number == 1
    assert np.isclose(signature.total_angle, 2.0 * np.pi)
    assert signature.endpoint_distance == 0.0


def test_activation_strata_counts_relu_regions_and_boundaries():
    activations = np.array(
        [
            [1.0, -1.0, 0.0],
            [2.0, 3.0, -1.0],
            [-1.0, -2.0, -3.0],
            [0.5, -0.5, 0.2],
        ],
        dtype=float,
    )

    strata = topoml.activation_strata(activations, threshold=0.0, boundary_tolerance=0.0)

    assert strata.stratum_counts == {"000": 1, "100": 1, "101": 1, "110": 1}
    assert strata.n_strata == 4
    assert np.isclose(strata.boundary_fraction, 1.0 / activations.size)


def test_finite_orbit_signature_summarizes_group_action():
    identity = np.eye(2)
    reflect_x = np.array([[-1.0, 0.0], [0.0, 1.0]])
    reflect_y = np.array([[1.0, 0.0], [0.0, -1.0]])
    rotate_180 = np.array([[-1.0, 0.0], [0.0, -1.0]])

    orbit = topoml.finite_orbit_signature(
        np.array([1.0, 0.0]),
        [identity, reflect_x, reflect_y, rotate_180],
    )

    assert orbit.orbit_size == 2
    assert orbit.stabilizer_count == 2
    assert np.isclose(orbit.orbit_diameter, 2.0)
    assert orbit.quotient_representative == (-1.0, 0.0)


def test_equivariance_residual_detects_invariance_violation():
    points = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=float)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    invariant = topoml.equivariance_residual(
        points,
        lambda x: x.sum(axis=1, keepdims=True),
        {"swap": swap},
    )
    variant = topoml.equivariance_residual(
        points,
        lambda x: x[:, :1],
        {"swap": swap},
    )

    assert invariant.action_residuals == {"swap": 0.0}
    assert invariant.max_residual == 0.0
    assert variant.action_residuals["swap"] > 0.0


def test_scott_fixed_point_converges_monotone_reachability():
    adjacency = np.array(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=bool,
    )

    def update(reached):
        return (reached @ adjacency) | reached

    result = topoml.scott_fixed_point(
        update,
        np.array([True, False, False, False]),
        join=np.logical_or,
    )

    assert result.fixed_point.tolist() == [True, True, True, True]
    assert result.converged
    assert result.steps == 3
    assert result.residual == 0.0


def test_weak_convergence_residual_separates_probe_from_strong_norm():
    sequence = np.array([[0.0, 10.0], [0.5, -10.0], [1.0, 10.0]], dtype=float)
    limit = np.array([1.0, 0.0], dtype=float)
    probes = np.array([[1.0, 0.0]], dtype=float)

    report = topoml.weak_convergence_residual(sequence, limit, probes)

    assert report.probe_residuals.tolist() == [0.0]
    assert report.max_residual == 0.0
    assert report.mean_residual == 0.0
    assert report.strong_residual == 10.0


def test_finite_topology_signature_checks_open_set_axioms_and_separation():
    signature = topoml.finite_topology_signature(
        universe={"a", "b"},
        open_sets=[set(), {"a"}, {"a", "b"}],
    )
    indiscrete = topoml.finite_topology_signature(
        universe={"a", "b"},
        open_sets=[set(), {"a", "b"}],
    )

    assert signature.is_topology
    assert signature.n_open_sets == 3
    assert signature.is_connected
    assert signature.is_t0
    assert not signature.is_t1
    assert not indiscrete.is_t0


def test_dynamical_signature_finds_loss_critical_events_and_recurrence():
    values = np.array([3.0, 2.0, 1.2, 1.6, 1.1, 1.1, 1.4], dtype=float)

    signature = topoml.dynamical_signature(values, recurrence_tolerance=0.0)

    assert signature.critical_indices == (2, 3, 5)
    assert np.isclose(signature.descent_fraction, 3.0 / 6.0)
    assert signature.plateau_count == 1
    assert signature.recurrence_count == 1
    assert signature.final_drift == -1.6


def test_braid_crossing_signature_records_adjacent_swaps():
    strands = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.0], [0.5, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=float,
    )

    signature = topoml.braid_crossing_signature(strands)

    assert signature.crossing_count == 1
    assert signature.braid_word == ("sigma1",)
    assert signature.pair_counts == {"0-1": 1}


def test_mesh_euler_characteristic_summarizes_low_dimensional_surface():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    edges = [(0, 1), (1, 2), (0, 2)]
    faces = [(0, 1, 2)]

    signature = topoml.mesh_euler_characteristic(vertices, edges, faces)

    assert signature.vertices == 3
    assert signature.edges == 3
    assert signature.faces == 1
    assert signature.euler_characteristic == 1
    assert not signature.closed_orientable
    assert signature.genus is None


def test_topology_prototype_contracts_are_exported():
    exported = set(topoml.__all__)

    assert {"Cover", "MapperGraph", "SheafResidual"}.issubset(exported)
    assert {
        "PathHomotopySignature",
        "StratumSignature",
        "OrbitSignature",
        "EquivarianceResidual",
        "ScottFixedPointResult",
        "WeakConvergenceResidual",
        "BraidCrossingSignature",
        "DynamicalSignature",
        "FiniteTopologySignature",
        "MeshEulerSignature",
    }.issubset(exported)
    assert {
        "braid_crossing_signature",
        "dynamical_signature",
        "path_homotopy_signature",
        "activation_strata",
        "finite_orbit_signature",
        "finite_topology_signature",
        "equivariance_residual",
        "mesh_euler_characteristic",
        "scott_fixed_point",
        "weak_convergence_residual",
    }.issubset(exported)
