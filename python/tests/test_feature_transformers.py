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
