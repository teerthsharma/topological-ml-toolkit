from __future__ import annotations

import argparse
import json
from pathlib import Path

import topoml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/tutorial-dashboard.html"))
    args = parser.parse_args()

    dataset = topoml.make_noisy_circle(n_samples=32, noise=0.0, random_state=7)
    diagram = topoml.persistent_homology(dataset.points, max_dim=1, max_radius=2.0)
    features = topoml.PHFeaturizer(max_dim=1, radii=[0.0, 0.45], homology_dims=[0, 1]).fit_transform(
        [dataset.points]
    )
    written = topoml.write_dashboard(
        args.out,
        title="Point-cloud topology tutorial",
        diagram=diagram,
        feature_matrix=features,
        metadata={"dataset": dataset.name, "expected_betti": dataset.expected_betti},
    )
    payload = {
        "dashboard": str(written),
        "bytes": written.stat().st_size,
        "dataset": dataset.name,
        "claim_scope": "tutorial smoke for static dashboard export",
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
