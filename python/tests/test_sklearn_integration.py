from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import topoml


def _clouds() -> list[np.ndarray]:
    return [
        np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 0.0]], dtype=float),
        np.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], dtype=float),
        np.array([[1.0, 1.0], [1.1, 1.0], [8.0, 1.0]], dtype=float),
        np.array([[1.0, 1.0], [1.2, 1.0], [1.4, 1.0]], dtype=float),
    ]


def test_topoml_import_does_not_import_sklearn() -> None:
    code = "import json, sys, topoml; print(json.dumps('sklearn' in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert result.stdout.strip() == "false"


def test_make_sklearn_pipeline_is_lazy_public_api() -> None:
    assert "make_sklearn_pipeline" in topoml.__all__
    assert "SklearnUnavailableError" in topoml.__all__


def test_make_sklearn_pipeline_fits_real_classifier_when_installed() -> None:
    pytest.importorskip("sklearn")
    from sklearn.tree import DecisionTreeClassifier

    labels = np.array(["separated", "connected", "separated", "connected"], dtype=object)
    pipeline = topoml.make_sklearn_pipeline(
        DecisionTreeClassifier(random_state=0),
        radii=[0.0, 0.15, 1.0],
        max_dim=0,
    )

    pipeline.fit(_clouds(), labels)

    assert pipeline.predict(_clouds()).tolist() == labels.tolist()
