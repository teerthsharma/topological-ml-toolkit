from __future__ import annotations

import json
import subprocess
import sys

import pytest

import topoml


def test_topoml_import_does_not_import_triton_runtime_stack() -> None:
    code = """
import json
import sys
import topoml
print(json.dumps({name: name in sys.modules for name in ["torch", "triton"]}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == {"torch": False, "triton": False}


def test_triton_runtime_status_is_exported_without_loading_heavy_stack() -> None:
    assert "TritonRuntimeStatus" in topoml.__all__
    assert "TritonSchedule" in topoml.__all__
    assert "TritonScheduleBuilder" in topoml.__all__
    assert "triton_runtime_status" in topoml.__all__
    assert "triton_pairwise_l2" in topoml.__all__


def test_triton_schedule_builder_selects_topology_landmarks_with_baselines() -> None:
    keys = [
        [0.0, 0.0],
        [0.1, 0.0],
        [2.0, 0.0],
        [2.1, 0.0],
        [5.0, 0.0],
        [5.1, 0.0],
    ]

    builder = topoml.TritonScheduleBuilder(
        budget=4,
        sink_tokens=1,
        local_window=1,
        landmark_count=2,
        random_state=7,
    )
    schedule = builder.build(keys, query_index=5)

    assert schedule.selected_key_indices == (0, 3, 4, 5)
    assert schedule.dense_baseline_indices == (0, 1, 2, 3, 4, 5)
    assert schedule.local_baseline_indices == (0, 3, 4, 5)
    assert len(schedule.random_baseline_indices) == len(schedule.selected_key_indices)
    assert schedule.budget == 4
    assert schedule.query_index == 5
    assert schedule.strategy == "sink_local_landmark"
    assert schedule.budget_unit == "selected_keys"
    assert schedule.claim_scope == "schedule construction only; no Triton kernel or sparse-attention speedup claim"


def test_triton_schedule_builder_rejects_invalid_budget() -> None:
    builder = topoml.TritonScheduleBuilder(budget=0)

    with pytest.raises(ValueError, match="budget"):
        builder.build([[0.0], [1.0]], query_index=1)


def test_triton_pairwise_l2_matches_torch_cdist_when_cuda_available() -> None:
    status = topoml.triton_runtime_status()
    if not status.available:
        pytest.skip(status.message)

    import torch

    points = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [3.0, 4.0, 1.0],
            [6.0, 8.0, 2.0],
            [1.0, 2.0, 3.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    observed = topoml.triton_pairwise_l2(points, block=16)
    expected = torch.cdist(points, points)

    assert observed.device.type == "cuda"
    assert observed.dtype == torch.float32
    assert torch.allclose(observed, expected, rtol=1e-5, atol=1e-5)
