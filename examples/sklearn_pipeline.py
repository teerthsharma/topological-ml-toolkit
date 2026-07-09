from __future__ import annotations

import json

import topoml


def main() -> None:
    dataset = topoml.make_cluster_bridge()
    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        payload = {
            "dataset": dataset.name,
            "sklearn_available": False,
            "claim_scope": "sklearn tutorial requires optional scikit-learn dependency",
        }
        print(json.dumps(payload, sort_keys=True))
        return

    pipeline = topoml.make_sklearn_pipeline(
        DecisionTreeClassifier(random_state=0),
        radii=[0.0, 0.15, 1.0],
        max_dim=0,
    )
    clouds = [
        dataset.points[:2],
        dataset.points[:2] + 0.05,
        dataset.points[[0, 2]],
    ]
    labels = ["near", "near", "far"]
    pipeline.fit(clouds, labels)
    predicted = pipeline.predict(clouds)
    payload = {
        "dataset": dataset.name,
        "sklearn_available": True,
        "steps": [name for name, _step in pipeline.steps],
        "predicted": predicted.tolist(),
        "claim_scope": "tutorial smoke for sklearn Pipeline wiring, not model quality",
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
