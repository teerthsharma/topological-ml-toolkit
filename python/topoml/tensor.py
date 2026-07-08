from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TensorBundleSpec:
    """Runtime descriptor for a vector/tensor bundle used by topology features.

    This is inspired by DirectSum-style TensorBundle interoperability: operations
    on elements from different spaces first construct a compatible ambient space,
    then coerce both inputs into that space. Python cannot make those choices at
    compile time, so the descriptor keeps the rule explicit and testable.
    """

    basis: tuple[str, ...]
    metric: tuple[float, ...]
    tangent_order: int = 0
    tangent_variables: int = 0
    dual: bool = False

    @classmethod
    def from_signature(
        cls,
        signature: str,
        *,
        prefix: str = "v",
        tangent_order: int = 0,
        tangent_variables: int = 0,
    ) -> "TensorBundleSpec":
        metric = []
        basis = []
        for index, char in enumerate(signature, start=1):
            if char == "+":
                value = 1.0
            elif char == "-":
                value = -1.0
            elif char == "0":
                value = 0.0
            else:
                raise ValueError("signature must contain only '+', '-', or '0'")
            metric.append(value)
            basis.append(f"{prefix}{index}")
        return cls(
            basis=tuple(basis),
            metric=tuple(metric),
            tangent_order=tangent_order,
            tangent_variables=tangent_variables,
        )

    @property
    def rank(self) -> int:
        return len(self.basis)

    def direct_sum(self, other: "TensorBundleSpec") -> "TensorBundleSpec":
        basis = list(self.basis)
        metric = list(self.metric)
        for name, value in zip(other.basis, other.metric, strict=True):
            next_name = name
            suffix = 2
            while next_name in basis:
                next_name = f"{name}_{suffix}"
                suffix += 1
            basis.append(next_name)
            metric.append(value)
        return TensorBundleSpec(
            basis=tuple(basis),
            metric=tuple(metric),
            tangent_order=max(self.tangent_order, other.tangent_order),
            tangent_variables=max(self.tangent_variables, other.tangent_variables),
            dual=self.dual and other.dual,
        )

    def union(self, other: "TensorBundleSpec") -> "TensorBundleSpec":
        basis = list(self.basis)
        metric = list(self.metric)
        positions = {name: index for index, name in enumerate(basis)}
        for name, value in zip(other.basis, other.metric, strict=True):
            if name in positions:
                existing = metric[positions[name]]
                if existing != value:
                    raise ValueError(f"incompatible metric for shared basis {name!r}")
                continue
            positions[name] = len(basis)
            basis.append(name)
            metric.append(value)
        return TensorBundleSpec(
            basis=tuple(basis),
            metric=tuple(metric),
            tangent_order=max(self.tangent_order, other.tangent_order),
            tangent_variables=max(self.tangent_variables, other.tangent_variables),
            dual=self.dual and other.dual,
        )

    def intersection(self, other: "TensorBundleSpec") -> "TensorBundleSpec":
        other_metric = dict(zip(other.basis, other.metric, strict=True))
        basis = []
        metric = []
        for name, value in zip(self.basis, self.metric, strict=True):
            if name in other_metric:
                if other_metric[name] != value:
                    raise ValueError(f"incompatible metric for shared basis {name!r}")
                basis.append(name)
                metric.append(value)
        return TensorBundleSpec(
            basis=tuple(basis),
            metric=tuple(metric),
            tangent_order=min(self.tangent_order, other.tangent_order),
            tangent_variables=min(self.tangent_variables, other.tangent_variables),
            dual=self.dual and other.dual,
        )

    def contains(self, other: "TensorBundleSpec") -> bool:
        metric = dict(zip(self.basis, self.metric, strict=True))
        return all(metric.get(name) == value for name, value in zip(other.basis, other.metric, strict=True))

    def dual_space(self) -> "TensorBundleSpec":
        return TensorBundleSpec(
            basis=tuple(f"{name}_dual" for name in self.basis),
            metric=tuple(-value for value in self.metric),
            tangent_order=self.tangent_order,
            tangent_variables=self.tangent_variables,
            dual=not self.dual,
        )


@dataclass(frozen=True)
class TensorAlgebraElement:
    """Dense coordinate element tied to a `TensorBundleSpec`."""

    bundle: TensorBundleSpec
    coordinates: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=float)
        if coords.shape != (self.bundle.rank,):
            raise ValueError("coordinates must have shape (bundle.rank,)")
        object.__setattr__(self, "coordinates", coords)

    def coerce(self, target: TensorBundleSpec) -> "TensorAlgebraElement":
        if not target.contains(self.bundle):
            raise ValueError("target bundle must contain the element bundle")
        target_positions = {name: index for index, name in enumerate(target.basis)}
        values = np.zeros(target.rank, dtype=float)
        for source_index, name in enumerate(self.bundle.basis):
            values[target_positions[name]] = self.coordinates[source_index]
        return TensorAlgebraElement(target, values)


def interop_bundle(
    left: TensorBundleSpec,
    right: TensorBundleSpec,
    *,
    mode: str = "union",
) -> TensorBundleSpec:
    """Return the compatible ambient bundle for a mixed-space operation."""

    if mode == "union":
        return left.union(right)
    if mode == "direct_sum":
        return left.direct_sum(right)
    raise ValueError("mode must be 'union' or 'direct_sum'")


def interop_add(
    left: TensorAlgebraElement,
    right: TensorAlgebraElement,
    *,
    mode: str = "union",
) -> TensorAlgebraElement:
    """Add two tensor elements after coercing them to a compatible bundle."""

    bundle = interop_bundle(left.bundle, right.bundle, mode=mode)
    coerced_left = left.coerce(bundle)
    coerced_right = right.coerce(bundle)
    return TensorAlgebraElement(bundle, coerced_left.coordinates + coerced_right.coordinates)


def bundle_metric_matrix(bundle: TensorBundleSpec) -> np.ndarray:
    """Return the diagonal metric matrix associated with a bundle spec."""

    return np.diag(np.asarray(bundle.metric, dtype=float))


def tensor_bundle_signature_features(bundle: TensorBundleSpec) -> dict[str, float]:
    """Expose bundle structure as numeric ML metadata features."""

    metric = np.asarray(bundle.metric, dtype=float)
    return {
        "rank": float(bundle.rank),
        "positive_metric_axes": float(np.sum(metric > 0.0)),
        "negative_metric_axes": float(np.sum(metric < 0.0)),
        "degenerate_metric_axes": float(np.sum(metric == 0.0)),
        "tangent_order": float(bundle.tangent_order),
        "tangent_variables": float(bundle.tangent_variables),
        "is_dual": float(bundle.dual),
    }
