from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .features import PHFeaturizer


class TopologyAugmenter:
    """Append persistent-homology features to ordinary ML feature matrices."""

    def __init__(
        self,
        *,
        radii: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
        max_dim: int = 1,
        homology_dims: Sequence[int] | None = None,
        include_original: bool = True,
    ):
        self.radii = tuple(float(radius) for radius in radii)
        self.max_dim = int(max_dim)
        self.homology_dims = None if homology_dims is None else tuple(int(dim) for dim in homology_dims)
        self.include_original = include_original

    def fit(
        self,
        clouds: Iterable[np.ndarray],
        y: object | None = None,
        *,
        base_features: np.ndarray | None = None,
    ) -> "TopologyAugmenter":
        del y
        cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
        self.featurizer_ = PHFeaturizer(max_dim=self.max_dim, radii=self.radii, homology_dims=self.homology_dims)
        self.featurizer_.fit(cached)
        topo = self.featurizer_.transform(cached)
        base_width = 0 if base_features is None else _as_2d(base_features).shape[1]
        self.n_features_out_ = (base_width if self.include_original else 0) + topo.shape[1]
        self.feature_names_out_ = tuple(
            [f"x{i}" for i in range(base_width)] if self.include_original else []
        ) + tuple(f"topology_{name}" for name in self.featurizer_.get_feature_names_out())
        return self

    def transform(self, clouds: Iterable[np.ndarray], *, base_features: np.ndarray | None = None) -> np.ndarray:
        if not hasattr(self, "featurizer_"):
            raise RuntimeError("call fit before transform")
        cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
        topo = self.featurizer_.transform(cached)
        if not self.include_original:
            return topo
        if base_features is None:
            return topo
        base = _as_2d(base_features)
        if base.shape[0] != topo.shape[0]:
            raise ValueError("base_features row count must match clouds")
        return np.hstack([base, topo])

    def fit_transform(
        self,
        clouds: Iterable[np.ndarray],
        y: object | None = None,
        *,
        base_features: np.ndarray | None = None,
    ) -> np.ndarray:
        cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
        return self.fit(cached, y, base_features=base_features).transform(cached, base_features=base_features)

    def get_feature_names_out(self) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise RuntimeError("call fit before requesting feature names")
        return np.asarray(self.feature_names_out_, dtype=object)


def topological_sample_weights(
    clouds: Iterable[np.ndarray],
    *,
    radii: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    max_dim: int = 1,
    homology_dims: Sequence[int] | None = None,
) -> np.ndarray:
    """Return mean-one sample weights from Betti-curve complexity.

    The weight is a deliberately simple training prior: examples whose topology
    changes more across the chosen radii receive higher weight. This is useful
    for pretraining or tabular baselines, but it is not a claim of optimality.
    """

    cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
    if not cached:
        return np.asarray([], dtype=float)
    features = PHFeaturizer(max_dim=max_dim, radii=radii, homology_dims=homology_dims).fit_transform(cached)
    variation = np.sum(np.abs(np.diff(features, axis=1)), axis=1) if features.shape[1] > 1 else 0.0
    mass = np.sum(np.abs(features), axis=1)
    weights = 1.0 + mass + variation
    mean = float(np.mean(weights))
    return weights / mean if mean > 0.0 else np.ones(len(cached), dtype=float)


@dataclass(frozen=True)
class _DecisionStump:
    feature_index: int
    threshold: float
    left_class: object
    right_class: object
    default_class: object

    def predict_row(self, row: np.ndarray) -> object:
        if not np.isfinite(row[self.feature_index]):
            return self.default_class
        return self.left_class if row[self.feature_index] <= self.threshold else self.right_class


class TopologyRandomForestClassifier:
    """Small topology-aware random forest baseline with no sklearn dependency.

    It trains an ensemble of weighted decision stumps on ordinary features
    augmented with Betti-curve features. The class is intentionally modest:
    it gives the repo an executable topology-first baseline while larger
    sklearn/PyTorch/TensorFlow integrations remain optional adapters.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 25,
        max_features: int | None = None,
        radii: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
        max_dim: int = 1,
        homology_dims: Sequence[int] | None = None,
        topology_weighted: bool = True,
        random_state: int = 0,
        thresholds_per_feature: int = 8,
    ):
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.radii = tuple(float(radius) for radius in radii)
        self.max_dim = int(max_dim)
        self.homology_dims = None if homology_dims is None else tuple(int(dim) for dim in homology_dims)
        self.topology_weighted = bool(topology_weighted)
        self.random_state = int(random_state)
        self.thresholds_per_feature = int(thresholds_per_feature)

    def fit(
        self,
        clouds: Iterable[np.ndarray],
        y: Sequence[object],
        *,
        base_features: np.ndarray | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> "TopologyRandomForestClassifier":
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if self.thresholds_per_feature <= 0:
            raise ValueError("thresholds_per_feature must be positive")

        cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
        labels = np.asarray(y, dtype=object)
        if labels.shape != (len(cached),):
            raise ValueError("y must contain one label per cloud")

        self.augmenter_ = TopologyAugmenter(
            radii=self.radii,
            max_dim=self.max_dim,
            homology_dims=self.homology_dims,
            include_original=True,
        )
        x = self.augmenter_.fit_transform(cached, base_features=base_features)
        weights = np.ones(labels.shape[0], dtype=float)
        if sample_weight is not None:
            weights *= np.asarray(sample_weight, dtype=float)
        if self.topology_weighted:
            weights *= topological_sample_weights(
                cached,
                radii=self.radii,
                max_dim=self.max_dim,
                homology_dims=self.homology_dims,
            )
        if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
            raise ValueError("sample weights must be finite and non-negative")

        self.classes_ = np.unique(labels)
        self.stumps_: list[_DecisionStump] = []
        rng = np.random.default_rng(self.random_state)
        feature_count = x.shape[1]
        subset_width = self.max_features or max(1, int(np.sqrt(feature_count)))
        subset_width = min(max(1, int(subset_width)), feature_count)
        probabilities = weights / weights.sum() if weights.sum() > 0.0 else None

        for _ in range(self.n_estimators):
            row_indices = rng.choice(len(labels), size=len(labels), replace=True, p=probabilities)
            feature_indices = rng.choice(feature_count, size=subset_width, replace=False)
            stump = _fit_best_stump(x[row_indices], labels[row_indices], weights[row_indices], feature_indices, self.classes_, self.thresholds_per_feature)
            self.stumps_.append(stump)
        return self

    def predict(self, clouds: Iterable[np.ndarray], *, base_features: np.ndarray | None = None) -> np.ndarray:
        if not hasattr(self, "stumps_"):
            raise RuntimeError("call fit before predict")
        cached = tuple(np.asarray(cloud, dtype=float) for cloud in clouds)
        x = self.augmenter_.transform(cached, base_features=base_features)
        predictions = []
        for row in x:
            votes = {label: 0 for label in self.classes_}
            for stump in self.stumps_:
                votes[stump.predict_row(row)] += 1
            predictions.append(max(self.classes_, key=lambda label: (votes[label], str(label))))
        return np.asarray(predictions, dtype=object)

    def score(
        self,
        clouds: Iterable[np.ndarray],
        y: Sequence[object],
        *,
        base_features: np.ndarray | None = None,
    ) -> float:
        labels = np.asarray(y, dtype=object)
        predicted = self.predict(clouds, base_features=base_features)
        if labels.shape != predicted.shape:
            raise ValueError("y shape must match predictions")
        return float(np.mean(predicted == labels))


def _as_2d(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim == 1:
        return matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("base_features must be a 1D or 2D array")
    return matrix


def _fit_best_stump(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    feature_indices: np.ndarray,
    classes: np.ndarray,
    thresholds_per_feature: int,
) -> _DecisionStump:
    default = _majority_class(y, weights, classes)
    best: tuple[float, _DecisionStump] | None = None
    for feature_index in feature_indices:
        column = x[:, feature_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0 or float(np.min(finite)) == float(np.max(finite)):
            continue
        quantiles = np.linspace(0.0, 1.0, thresholds_per_feature + 2)[1:-1]
        thresholds = np.unique(np.quantile(finite, quantiles))
        for threshold in thresholds:
            left = column <= threshold
            right = ~left
            if not left.any() or not right.any():
                continue
            left_class = _majority_class(y[left], weights[left], classes)
            right_class = _majority_class(y[right], weights[right], classes)
            loss = _weighted_gini(y[left], weights[left], classes) + _weighted_gini(y[right], weights[right], classes)
            stump = _DecisionStump(int(feature_index), float(threshold), left_class, right_class, default)
            if best is None or loss < best[0]:
                best = (loss, stump)
    if best is None:
        return _DecisionStump(0, float("inf"), default, default, default)
    return best[1]


def _majority_class(y: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> object:
    totals = {label: 0.0 for label in classes}
    for label, weight in zip(y, weights, strict=True):
        totals[label] += float(weight)
    return max(classes, key=lambda label: (totals[label], str(label)))


def _weighted_gini(y: np.ndarray, weights: np.ndarray, classes: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    impurity = 1.0
    for label in classes:
        probability = float(np.sum(weights[y == label])) / total
        impurity -= probability * probability
    return impurity * total
