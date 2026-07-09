# Topology Training Pipeline

Topological ML becomes useful when topology enters a normal training workflow
without forcing the whole project to become a topology project.

```mermaid
classDiagram
  class PHFeaturizer {
    +fit(clouds)
    +transform(clouds)
    +get_feature_names_out()
  }
  class TopologyAugmenter {
    +fit_transform(point_clouds, base_features)
  }
  class TopologyRandomForestClassifier {
    +fit(point_clouds, labels)
    +predict(point_clouds)
    +score(point_clouds, labels)
  }
  class TensorBundleSpec {
    +direct_sum(other)
    +union(other)
    +intersection(other)
  }
  PHFeaturizer --> TopologyAugmenter
  TopologyAugmenter --> TopologyRandomForestClassifier
  TensorBundleSpec --> TopologyAugmenter
```

## Active API

```python
augmenter = topoml.TopologyAugmenter(radii=[0.0, 0.25, 0.5], max_dim=1)
features = augmenter.fit_transform(point_clouds, base_features=tabular_features)
weights = topoml.topological_sample_weights(point_clouds, radii=[0.0, 0.25, 0.5])
model = topoml.TopologyRandomForestClassifier(radii=[0.0, 0.25, 0.5])
model.fit(point_clouds, labels, base_features=tabular_features, sample_weight=weights)
```

```mermaid
journey
  title Data Scientist Path
  section Inspect
    Build point clouds: 4: User
    Plot barcode and Betti curve: 5: User
  section Train
    Append topology features: 5: User
    Fit topology random forest: 4: User
  section Verify
    Compare against simple baselines: 5: User
    Publish claim boundary: 5: User
```

## Claim Boundary

The current training surface is an executable baseline for tabular experiments.
It is not a replacement for scikit-learn, PyTorch, TensorFlow, or XGBoost. Use it
to prove that topology features add signal before moving the same ideas into
larger training stacks.
