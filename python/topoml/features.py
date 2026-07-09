from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .core import PersistenceDiagram, persistent_homology
from .topology import TopologySignature


class PHFeaturizer:
    """sklearn-style persistent-homology feature transformer.

    The transformer converts each point cloud into a fixed-width Betti curve:
    for every requested homology dimension and radius, it records the active
    Betti number. The output is a normal NumPy matrix suitable for sklearn,
    PyTorch, TensorFlow, or tabular feature pipelines.
    """

    def __init__(
        self,
        max_dim: int = 1,
        radii: Sequence[float] | None = None,
        homology_dims: Sequence[int] | None = None,
    ):
        self.max_dim = max_dim
        self.radii = tuple(float(radius) for radius in (radii or (0.0, 0.25, 0.5, 1.0)))
        self.homology_dims = tuple(int(dim) for dim in (homology_dims or range(max_dim + 1)))
        self.is_fitted_ = False

    def fit(self, clouds: Iterable[np.ndarray], y: object | None = None) -> "PHFeaturizer":
        del y
        self._validate_params()
        self.n_features_out_ = len(self.radii) * len(self.homology_dims)
        self.feature_names_out_ = tuple(
            f"beta{dim}@{radius:g}" for dim in self.homology_dims for radius in self.radii
        )
        self.is_fitted_ = True
        return self

    def transform(self, clouds: Iterable[np.ndarray]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("call fit before transform")

        rows: list[list[float]] = []
        max_radius = max(self.radii) if self.radii else 0.0
        for cloud in clouds:
            diagram = persistent_homology(cloud, max_dim=self.max_dim, max_radius=max_radius)
            row: list[float] = []
            for dim in self.homology_dims:
                for radius in self.radii:
                    betti = diagram.betti_at(radius)
                    row.append(float(getattr(betti, f"beta{dim}")))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, clouds: Iterable[np.ndarray], y: object | None = None) -> np.ndarray:
        return self.fit(clouds, y).transform(clouds)

    def get_params(self, deep: bool = True) -> dict[str, object]:
        del deep
        return {
            "max_dim": self.max_dim,
            "radii": self.radii,
            "homology_dims": self.homology_dims,
        }

    def set_params(self, **params: object) -> "PHFeaturizer":
        valid = {"max_dim", "radii", "homology_dims"}
        unknown = set(params) - valid
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown PHFeaturizer parameter(s): {names}")
        if "max_dim" in params:
            self.max_dim = int(params["max_dim"])
        if "radii" in params:
            self.radii = tuple(float(radius) for radius in params["radii"])  # type: ignore[union-attr]
        if "homology_dims" in params:
            value = params["homology_dims"]
            self.homology_dims = tuple(int(dim) for dim in value) if value is not None else tuple(range(self.max_dim + 1))  # type: ignore[arg-type]
        self.is_fitted_ = False
        return self

    def get_feature_names_out(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("call fit before requesting feature names")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _validate_params(self) -> None:
        if self.max_dim < 0 or self.max_dim > 2:
            raise ValueError("max_dim must be 0, 1, or 2")
        if not self.radii:
            raise ValueError("radii must contain at least one value")
        if any(radius < 0 or not np.isfinite(radius) for radius in self.radii):
            raise ValueError("radii must be finite non-negative values")
        if any(dim < 0 or dim > self.max_dim for dim in self.homology_dims):
            raise ValueError("homology_dims must be between 0 and max_dim")


class BettiCurve:
    """Encode persistence diagrams as fixed-width Betti curve samples."""

    def __init__(self, radii: Sequence[float], homology_dims: Sequence[int] = (0, 1)):
        self.radii = tuple(float(radius) for radius in radii)
        self.homology_dims = tuple(int(dim) for dim in homology_dims)
        self.is_fitted_ = False

    def fit(self, diagrams: Iterable[PersistenceDiagram], y: object | None = None) -> "BettiCurve":
        del diagrams, y
        self._validate_params()
        self.n_features_out_ = len(self.radii) * len(self.homology_dims)
        self.feature_names_out_ = tuple(
            f"beta{dim}@{radius:g}" for dim in self.homology_dims for radius in self.radii
        )
        self.is_fitted_ = True
        return self

    def transform(self, diagrams: Iterable[PersistenceDiagram]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("call fit before transform")
        rows: list[list[float]] = []
        for diagram in diagrams:
            row: list[float] = []
            for dim in self.homology_dims:
                for radius in self.radii:
                    betti = diagram.betti_at(radius)
                    row.append(float(getattr(betti, f"beta{dim}")))
            rows.append(row)
        return np.asarray(rows, dtype=float)

    def fit_transform(self, diagrams: Iterable[PersistenceDiagram], y: object | None = None) -> np.ndarray:
        return self.fit(diagrams, y).transform(diagrams)

    def get_feature_names_out(self) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("call fit before requesting feature names")
        return np.asarray(self.feature_names_out_, dtype=object)

    def _validate_params(self) -> None:
        if not self.radii:
            raise ValueError("radii must contain at least one value")
        if any(radius < 0 or not np.isfinite(radius) for radius in self.radii):
            raise ValueError("radii must be finite non-negative values")
        if any(dim < 0 or dim > 2 for dim in self.homology_dims):
            raise ValueError("homology_dims must be 0, 1, or 2")


class PersistenceImage:
    """Encode finite persistence pairs as a fixed-width image-like vector."""

    def __init__(
        self,
        width: int = 16,
        height: int = 16,
        sigma: float = 0.1,
        birth_range: tuple[float, float] | None = None,
        persistence_range: tuple[float, float] | None = None,
    ):
        self.width = width
        self.height = height
        self.sigma = sigma
        self.birth_range = birth_range
        self.persistence_range = persistence_range
        self.is_fitted_ = False

    def fit(self, diagrams: Iterable[PersistenceDiagram], y: object | None = None) -> "PersistenceImage":
        del y
        self._validate_params()
        pairs = [
            (pair.birth, pair.death - pair.birth)
            for diagram in diagrams
            for pair in diagram.finite_pairs()
            if pair.death is not None and pair.death > pair.birth
        ]
        if pairs:
            births = [birth for birth, _persistence in pairs]
            persistences = [persistence for _birth, persistence in pairs]
            self.birth_range_ = self.birth_range or (float(min(births)), float(max(births)))
            self.persistence_range_ = self.persistence_range or (0.0, float(max(persistences)))
        else:
            self.birth_range_ = self.birth_range or (0.0, 1.0)
            self.persistence_range_ = self.persistence_range or (0.0, 1.0)
        self.n_features_out_ = self.width * self.height
        self.is_fitted_ = True
        return self

    def transform(self, diagrams: Iterable[PersistenceDiagram]) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("call fit before transform")
        xs = np.linspace(self.birth_range_[0], self.birth_range_[1], self.width)
        ys = np.linspace(self.persistence_range_[0], self.persistence_range_[1], self.height)
        grid_x, grid_y = np.meshgrid(xs, ys)
        rows: list[np.ndarray] = []
        for diagram in diagrams:
            image = np.zeros((self.height, self.width), dtype=float)
            for pair in diagram.finite_pairs():
                if pair.death is None or pair.death <= pair.birth:
                    continue
                persistence = pair.death - pair.birth
                weight = persistence
                image += weight * np.exp(
                    -((grid_x - pair.birth) ** 2 + (grid_y - persistence) ** 2) / (2.0 * self.sigma**2)
                )
            rows.append(image.reshape(-1))
        return np.asarray(rows, dtype=float)

    def fit_transform(self, diagrams: Iterable[PersistenceDiagram], y: object | None = None) -> np.ndarray:
        cached = tuple(diagrams)
        return self.fit(cached, y).transform(cached)

    def _validate_params(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.sigma <= 0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be a finite positive value")


def point_cloud_signature(
    points: np.ndarray,
    *,
    radii: Sequence[float],
    max_dim: int = 1,
) -> TopologySignature:
    diagram = persistent_homology(points, max_dim=max_dim, max_radius=max(radii))
    curve = BettiCurve(radii=radii, homology_dims=range(max_dim + 1))
    values = curve.fit_transform([diagram])[0]
    return TopologySignature(
        kind="point_cloud",
        values={name: float(value) for name, value in zip(curve.get_feature_names_out(), values, strict=True)},
    )


def activation_signature(
    activations: np.ndarray,
    *,
    radii: Sequence[float],
    max_dim: int = 1,
) -> TopologySignature:
    tensor = np.asarray(activations, dtype=float)
    if tensor.ndim < 2:
        raise ValueError("activations must have at least two dimensions")
    cloud = tensor.reshape(-1, tensor.shape[-1])
    signature = point_cloud_signature(cloud, radii=radii, max_dim=max_dim)
    return TopologySignature(
        kind="activation",
        values={"samples": float(cloud.shape[0]), "features": float(cloud.shape[1]), **signature.values},
    )
