from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .core import persistent_homology


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
