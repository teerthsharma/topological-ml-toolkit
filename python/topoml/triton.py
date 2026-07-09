from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class TritonRuntimeStatus:
    torch_importable: bool
    triton_importable: bool
    cuda_available: bool
    available: bool
    message: str


class TritonBackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TritonSchedule:
    selected_key_indices: tuple[int, ...]
    dense_baseline_indices: tuple[int, ...]
    local_baseline_indices: tuple[int, ...]
    random_baseline_indices: tuple[int, ...]
    query_index: int
    budget: int
    budget_unit: str
    strategy: str
    claim_scope: str


@dataclass(frozen=True)
class TritonScheduleBuilder:
    """Build topology-guided candidate schedules for later Triton benchmarks.

    The builder is CPU-side and deterministic. It selects sink tokens, a local
    causal window, and farthest-point landmarks from key embeddings. The result
    also carries local-only and same-budget random baselines, because schedule
    construction is not meaningful without ablations.
    """

    budget: int
    sink_tokens: int = 1
    local_window: int = 32
    landmark_count: int = 8
    random_state: int = 0

    def build(self, key_embeddings: Sequence[Sequence[float]] | np.ndarray, *, query_index: int) -> TritonSchedule:
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.sink_tokens < 0:
            raise ValueError("sink_tokens must be non-negative")
        if self.local_window < 0:
            raise ValueError("local_window must be non-negative")
        if self.landmark_count < 0:
            raise ValueError("landmark_count must be non-negative")

        keys = np.asarray(key_embeddings, dtype=float)
        if keys.ndim != 2 or keys.shape[0] == 0:
            raise ValueError("key_embeddings must be a non-empty 2D array")
        if not np.all(np.isfinite(keys)):
            raise ValueError("key_embeddings must contain only finite values")
        n_keys = int(keys.shape[0])
        if query_index < 0 or query_index >= n_keys:
            raise ValueError("query_index is out of range")

        causal = np.arange(query_index + 1, dtype=int)
        selected: set[int] = set(causal[: min(self.sink_tokens, len(causal))].tolist())
        local_start = max(0, query_index - self.local_window + 1)
        selected.update(range(local_start, query_index + 1))

        remaining_budget = max(0, min(self.budget, len(causal)) - len(selected))
        if remaining_budget > 0 and self.landmark_count > 0:
            candidates = np.array([idx for idx in causal if int(idx) not in selected], dtype=int)
            landmarks = _farthest_point_landmarks(
                keys,
                candidates,
                count=min(self.landmark_count, remaining_budget),
                seed_index=min(self.sink_tokens, len(causal) - 1),
            )
            selected.update(landmarks)

        if len(selected) > self.budget:
            selected = set(sorted(selected)[: self.budget])

        chosen = tuple(sorted(selected))
        local_baseline = _local_baseline(causal, query_index=query_index, budget=len(chosen), sink_tokens=self.sink_tokens)
        random_baseline = _random_baseline(causal, budget=len(chosen), random_state=self.random_state)
        return TritonSchedule(
            selected_key_indices=chosen,
            dense_baseline_indices=tuple(int(idx) for idx in causal),
            local_baseline_indices=local_baseline,
            random_baseline_indices=random_baseline,
            query_index=query_index,
            budget=self.budget,
            budget_unit="selected_keys",
            strategy="sink_local_landmark",
            claim_scope="schedule construction only; no Triton kernel or sparse-attention speedup claim",
        )


def _farthest_point_landmarks(
    keys: np.ndarray,
    candidates: np.ndarray,
    *,
    count: int,
    seed_index: int,
) -> tuple[int, ...]:
    if count <= 0 or candidates.size == 0:
        return ()
    chosen: list[int] = []
    seed = int(seed_index)
    distances_to_chosen = np.linalg.norm(keys[candidates] - keys[seed], axis=1)
    for _ in range(count):
        if distances_to_chosen.size == 0:
            break
        position = int(np.argmax(distances_to_chosen))
        idx = int(candidates[position])
        chosen.append(idx)
        candidates = np.delete(candidates, position)
        distances_to_chosen = np.delete(distances_to_chosen, position)
        if candidates.size == 0:
            break
        next_distances = np.linalg.norm(keys[candidates] - keys[idx], axis=1)
        distances_to_chosen = np.minimum(distances_to_chosen, next_distances)
    return tuple(chosen)


def _local_baseline(causal: np.ndarray, *, query_index: int, budget: int, sink_tokens: int) -> tuple[int, ...]:
    selected: set[int] = set(causal[: min(max(0, sink_tokens), len(causal))].tolist())
    cursor = query_index
    while len(selected) < budget and cursor >= 0:
        selected.add(cursor)
        cursor -= 1
    return tuple(sorted(selected))


def _random_baseline(causal: np.ndarray, *, budget: int, random_state: int) -> tuple[int, ...]:
    rng = np.random.default_rng(random_state)
    if budget >= len(causal):
        return tuple(int(idx) for idx in causal)
    selected = rng.choice(causal, size=budget, replace=False)
    return tuple(sorted(int(idx) for idx in selected))


def triton_runtime_status() -> TritonRuntimeStatus:
    torch_spec = importlib.util.find_spec("torch")
    triton_spec = importlib.util.find_spec("triton")
    torch_importable = torch_spec is not None
    triton_importable = triton_spec is not None
    cuda_available = False

    if torch_importable:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            torch_importable = False

    if triton_importable:
        try:
            import triton  # noqa: F401
        except Exception:
            triton_importable = False

    available = torch_importable and triton_importable and cuda_available
    if available:
        message = "Triton runtime is available with torch, triton, and CUDA."
    else:
        missing = []
        if not torch_importable:
            missing.append("torch import")
        if not triton_importable:
            missing.append("triton import")
        if not cuda_available:
            missing.append("CUDA device")
        message = "Triton runtime unavailable; missing " + ", ".join(missing) + "."

    return TritonRuntimeStatus(
        torch_importable=torch_importable,
        triton_importable=triton_importable,
        cuda_available=cuda_available,
        available=available,
        message=message,
    )


def triton_pairwise_l2(points: Any, *, block: int = 1024) -> Any:
    status = triton_runtime_status()
    if not status.available:
        raise TritonBackendUnavailable(status.message)
    module = _load_triton_distance_module()
    return module.pairwise_l2(points, block=block)


def _load_triton_distance_module() -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    source = root / "backends" / "triton" / "topology_distance.py"
    spec = importlib.util.spec_from_file_location("topoml_triton_topology_distance", source)
    if spec is None or spec.loader is None:
        raise TritonBackendUnavailable(f"could not load Triton source at {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
