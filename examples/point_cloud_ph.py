from __future__ import annotations

import json

import topoml


def main() -> None:
    dataset = topoml.make_noisy_circle(n_samples=32, noise=0.0, random_state=7)
    diagram = topoml.persistent_homology(dataset.points, max_dim=1, max_radius=2.0)
    features = topoml.PHFeaturizer(max_dim=1, radii=[0.0, 0.25, 0.45, 1.0], homology_dims=[1]).fit_transform(
        [dataset.points]
    )
    betti = diagram.betti_at(0.45)
    payload = {
        "dataset": dataset.name,
        "points": list(dataset.points.shape),
        "betti_at_radius": {"radius": 0.45, "beta0": betti.beta0, "beta1": betti.beta1},
        "feature_shape": list(features.shape),
        "feature_names": ["beta1@0", "beta1@0.25", "beta1@0.45", "beta1@1"],
        "claim_scope": "tutorial smoke for a deterministic synthetic circle fixture",
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
