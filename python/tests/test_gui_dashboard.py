import numpy as np

import topoml


def test_write_dashboard_creates_self_contained_html(tmp_path):
    points = np.array([[0.0, 0.0], [0.2, 0.0], [1.0, 0.0]], dtype=float)
    diagram = topoml.persistent_homology(points, max_dim=0, max_radius=2.0)
    features = topoml.PHFeaturizer(max_dim=0, radii=[0.0, 0.5]).fit_transform([points])
    out = tmp_path / "dashboard.html"

    written = topoml.write_dashboard(
        out,
        title="Topology inspection",
        diagram=diagram,
        feature_matrix=features,
        metadata={"dataset": "unit"},
    )

    html = written.read_text(encoding="utf-8")
    assert written == out
    assert "<!doctype html>" in html.lower()
    assert "Topology inspection" in html
    assert "Persistence Diagram" in html
    assert "Feature Matrix" in html
    assert "dataset" in html
    assert "plotly" not in html.lower()


def test_dashboard_exporter_is_public_api():
    assert "write_dashboard" in topoml.__all__
