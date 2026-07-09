from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .features import PHFeaturizer


class SklearnUnavailableError(RuntimeError):
    pass


def make_sklearn_pipeline(
    estimator: Any,
    *,
    radii: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    max_dim: int = 1,
    homology_dims: Sequence[int] | None = None,
    topology_step_name: str = "topology",
    estimator_step_name: str = "estimator",
) -> Any:
    """Build a sklearn Pipeline with PHFeaturizer followed by an estimator.

    scikit-learn remains optional. Importing topoml does not import sklearn; this
    function imports it only when the caller asks for a sklearn Pipeline.
    """

    try:
        from sklearn.pipeline import Pipeline
    except Exception as exc:  # pragma: no cover - depends on optional install state
        raise SklearnUnavailableError("scikit-learn is required for make_sklearn_pipeline") from exc
    featurizer = PHFeaturizer(max_dim=max_dim, radii=radii, homology_dims=homology_dims)
    return Pipeline([(topology_step_name, featurizer), (estimator_step_name, estimator)])
