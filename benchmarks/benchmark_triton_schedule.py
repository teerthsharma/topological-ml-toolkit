from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import topoml


def run_benchmark() -> dict:
    keys = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [2.0, 0.0],
            [2.1, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
        ],
        dtype=float,
    )
    builder = topoml.TritonScheduleBuilder(
        budget=4,
        sink_tokens=1,
        local_window=1,
        landmark_count=2,
        random_state=7,
    )
    start = time.perf_counter()
    schedule = builder.build(keys, query_index=5)
    schedule_build_ms = (time.perf_counter() - start) * 1000.0
    return {
        "selected_key_indices": list(schedule.selected_key_indices),
        "dense_baseline_indices": list(schedule.dense_baseline_indices),
        "local_baseline_indices": list(schedule.local_baseline_indices),
        "random_baseline_indices": list(schedule.random_baseline_indices),
        "budget": schedule.budget,
        "budget_unit": schedule.budget_unit,
        "strategy": schedule.strategy,
        "schedule_build_ms": round(schedule_build_ms, 6),
        "claim_scope": schedule.claim_scope,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/triton-schedule.json"))
    args = parser.parse_args()

    payload = run_benchmark()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
