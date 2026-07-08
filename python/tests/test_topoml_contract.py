import numpy as np

import topoml


def test_h0_barcode_tracks_cluster_merges():
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)

    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)

    assert diagram.betti_at(0.1).beta0 == 3
    assert diagram.betti_at(0.3).beta0 == 2
    assert diagram.betti_at(6.0).beta0 == 1


def test_time_delay_embedding_is_graph_ready():
    samples = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0], dtype=float)

    cloud = topoml.time_delay_embedding(samples, dimension=3, tau=1)

    assert cloud.shape == (4, 3)
    assert cloud[0].tolist() == [0.0, 1.0, 0.0]


def test_persistence_diagram_exports_plotly_trace_shape():
    points = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=10.0)

    trace = diagram.to_plotly_trace()

    assert trace["type"] == "scatter"
    assert trace["mode"] == "markers"
    assert trace["name"] == "H0 intervals"
    assert trace["x"] == [0.0, 0.0]
    assert trace["y"] == [0.2, 4.8]

