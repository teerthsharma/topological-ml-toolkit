from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _run_example(path: str, *args: str) -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(ROOT / path), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_point_cloud_tutorial_runs_and_reports_expected_loop() -> None:
    payload = _run_example("examples/point_cloud_ph.py")

    assert payload["dataset"] == "noisy_circle"
    assert payload["betti_at_radius"]["beta1"] == 1
    assert payload["feature_shape"] == [1, 4]


def test_sklearn_pipeline_tutorial_runs_with_optional_dependency_state() -> None:
    payload = _run_example("examples/sklearn_pipeline.py")

    assert payload["dataset"] == "cluster_bridge"
    assert payload["sklearn_available"] in {True, False}
    if payload["sklearn_available"]:
        assert payload["predicted"] == ["near", "near", "far"]


def test_dashboard_tutorial_writes_html(tmp_path: Path) -> None:
    out = tmp_path / "tutorial-dashboard.html"

    payload = _run_example("examples/dashboard_export.py", "--out", str(out))

    assert payload["dashboard"] == str(out)
    assert payload["bytes"] > 500
    assert out.exists()
