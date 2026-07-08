import numpy as np

import topoml


def test_tensor_bundle_union_direct_sum_and_interop_add():
    xy = topoml.TensorBundleSpec(("x", "y"), (1.0, 1.0))
    yz = topoml.TensorBundleSpec(("y", "z"), (1.0, -1.0), tangent_order=1, tangent_variables=2)

    ambient = topoml.interop_bundle(xy, yz)
    assert ambient.basis == ("x", "y", "z")
    assert ambient.metric == (1.0, 1.0, -1.0)
    assert ambient.tangent_order == 1
    assert ambient.tangent_variables == 2

    summed = topoml.interop_add(
        topoml.TensorAlgebraElement(xy, np.array([1.0, 2.0])),
        topoml.TensorAlgebraElement(yz, np.array([3.0, 4.0])),
    )
    assert summed.bundle == ambient
    assert summed.coordinates.tolist() == [1.0, 5.0, 4.0]

    direct_sum = xy.direct_sum(xy)
    assert direct_sum.basis == ("x", "y", "x_2", "y_2")


def test_tensor_bundle_signature_features_and_metric_matrix():
    bundle = topoml.TensorBundleSpec.from_signature("++-0", tangent_order=2, tangent_variables=1)

    features = topoml.tensor_bundle_signature_features(bundle)
    metric = topoml.bundle_metric_matrix(bundle)

    assert features == {
        "rank": 4.0,
        "positive_metric_axes": 2.0,
        "negative_metric_axes": 1.0,
        "degenerate_metric_axes": 1.0,
        "tangent_order": 2.0,
        "tangent_variables": 1.0,
        "is_dual": 0.0,
    }
    assert np.diag(metric).tolist() == [1.0, 1.0, -1.0, 0.0]


def test_topology_augmenter_appends_betti_features_to_base_features():
    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
    ]
    base = np.array([[10.0], [20.0]], dtype=float)

    augmenter = topoml.TopologyAugmenter(radii=[0.0, 0.15, 1.0], max_dim=0)
    features = augmenter.fit_transform(clouds, base_features=base)

    assert features.tolist() == [
        [10.0, 3.0, 2.0, 2.0],
        [20.0, 3.0, 3.0, 1.0],
    ]
    assert augmenter.get_feature_names_out().tolist() == [
        "x0",
        "topology_beta0@0",
        "topology_beta0@0.15",
        "topology_beta0@1",
    ]


def test_topological_sample_weights_are_positive_and_mean_one():
    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
    ]

    weights = topoml.topological_sample_weights(clouds, radii=[0.0, 0.15, 1.0], max_dim=0)

    assert weights.shape == (2,)
    assert np.all(weights > 0.0)
    assert np.isclose(weights.mean(), 1.0)


def test_topology_random_forest_classifier_fits_topology_features():
    clouds = [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
        np.array([[1.0, 1.0], [1.1, 1.0], [8.0, 1.0]], dtype=float),
        np.array([[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]], dtype=float),
    ]
    labels = np.array(["separated", "connected", "separated", "connected"], dtype=object)

    model = topoml.TopologyRandomForestClassifier(
        n_estimators=15,
        max_features=2,
        radii=[0.0, 0.15, 1.0],
        max_dim=0,
        random_state=7,
    )
    model.fit(clouds, labels)

    assert model.score(clouds, labels) == 1.0
    assert model.predict(clouds).tolist() == labels.tolist()
