"""Topological ML Toolkit public Python API."""

from .core import (
    BackendMetadata,
    BettiNumbers,
    PersistenceDiagram,
    PersistencePair,
    available_backends,
    persistent_homology,
    select_backend,
    time_delay_embedding,
)
from .backends import (
    BackendAdapter,
    BackendSelectionResult,
    BackendUnavailableError,
    backend_adapter,
    backend_adapters,
    select_backend_adapter,
)
from .features import PHFeaturizer
from .gui import write_dashboard
from .topology import (
    Cover,
    CoverCell,
    MapperGraph,
    NerveGraph,
    SheafResidual,
    mapper_graph,
    metric_cover,
    nerve_graph,
    sheaf_consistency_residual,
)

__all__ = [
    "BackendAdapter",
    "BackendMetadata",
    "BackendSelectionResult",
    "BackendUnavailableError",
    "BettiNumbers",
    "Cover",
    "CoverCell",
    "MapperGraph",
    "NerveGraph",
    "PHFeaturizer",
    "PersistenceDiagram",
    "PersistencePair",
    "SheafResidual",
    "available_backends",
    "backend_adapter",
    "backend_adapters",
    "mapper_graph",
    "metric_cover",
    "nerve_graph",
    "persistent_homology",
    "select_backend",
    "select_backend_adapter",
    "sheaf_consistency_residual",
    "time_delay_embedding",
    "write_dashboard",
]
