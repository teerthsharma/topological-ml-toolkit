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


def test_topology_prototype_contracts_are_exported():
    exported = set(topoml.__all__)

    assert {"Cover", "MapperGraph", "SheafResidual"}.issubset(exported)
