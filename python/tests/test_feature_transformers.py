import numpy as np
import pytest

import topoml


def test_ph_featurizer_exports_fixed_width_betti_curves():
    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
    ]

    features = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.15, 1.0]).fit_transform(clouds)

    assert features.shape == (2, 3)
    assert features.tolist() == [
        [3.0, 2.0, 2.0],
        [3.0, 3.0, 1.0],
    ]


def test_ph_featurizer_tracks_h1_when_requested():
    square = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    features = topoml.PHFeaturizer(max_dim=1, radii=[1.1], homology_dims=[0, 1]).fit_transform([square])

    assert features.shape == (1, 2)
    assert features[0, 0] == 1.0
    assert features[0, 1] == 1.0


def test_ph_featurizer_requires_fit_before_transform():
    featurizer = topoml.PHFeaturizer(max_dim=0, radii=[0.5])

    with pytest.raises(RuntimeError, match="fit before transform"):
        featurizer.transform([np.array([[0.0, 0.0]], dtype=float)])


def test_betti_curve_encodes_existing_diagrams():
    diagram = topoml.persistent_homology(
        np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float),
        max_dim=0,
        max_radius=10.0,
    )

    encoder = topoml.BettiCurve(radii=[0.0, 0.3, 6.0], homology_dims=[0])
    features = encoder.fit_transform([diagram])

    assert features.shape == (1, 3)
    assert features.tolist() == [[3.0, 2.0, 1.0]]
    assert encoder.get_feature_names_out().tolist() == ["beta0@0", "beta0@0.3", "beta0@6"]


def test_persistence_image_has_stable_shape_and_positive_mass():
    diagram = topoml.persistent_homology(
        np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float),
        max_dim=0,
        max_radius=10.0,
    )

    image = topoml.PersistenceImage(width=4, height=3, sigma=0.2).fit_transform([diagram])

    assert image.shape == (1, 12)
    assert image.sum() > 0.0
    assert np.isfinite(image).all()


def test_topology_signature_summarizes_point_cloud_graph_and_activation():
    point_signature = topoml.point_cloud_signature(
        np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 0.0]], dtype=float),
        radii=[0.0, 0.3],
        max_dim=0,
    )
    graph_signature = topoml.graph_signature(
        np.array(
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
            dtype=float,
        )
    )
    activation_signature = topoml.activation_signature(
        np.array(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0]],
            ],
            dtype=float,
        ),
        radii=[0.0, 1.1],
        max_dim=0,
    )

    assert point_signature.kind == "point_cloud"
    assert point_signature.values["beta0@0.3"] == 2.0
    assert graph_signature.values == {"nodes": 3.0, "edges": 3.0, "components": 1.0, "cycle_rank": 1.0}
    assert activation_signature.kind == "activation"
    assert activation_signature.values["samples"] == 4.0
    assert activation_signature.values["beta0@1.1"] == 1.0
