# Topology Family Coverage Matrix

This page is the honest coverage dashboard for the objective taxonomy. It
separates documentation, prototypes, stable APIs, and benchmark evidence.

| Family | Current status | Active API | Benchmark status | Next deliverable |
| --- | --- | --- | --- | --- |
| Point-set / general topology | Prototype | [`finite_topology_signature`](../reference/api.md#prototype-topology-diagnostics) | E2E finite topology check | Tutorial on continuity, compactness, connectedness, and finite covers in data pipelines |
| Metric, uniform, and proximity spaces | Prototype | [`metric_cover`, `nerve_graph`](../reference/api.md#prototype-topology-diagnostics) | E2E cover and nerve checks | [Cover-routing gallery](../gallery/covers-nerves-routing.md) with same-budget random/locality baseline |
| Domain theory / Scott topology | Prototype | [`scott_fixed_point`](../reference/api.md#prototype-topology-diagnostics) | E2E monotone fixed-point check | Monotone dataflow and scheduler invariant design |
| Homotopy theory | Prototype | [`path_homotopy_signature`](../reference/api.md#prototype-topology-diagnostics) | E2E winding-number check | Trajectory path-class tutorial for optimization and systems state |
| Cohomology beyond Betti counts | Prototype | [`sheaf_consistency_residual`](../reference/api.md#prototype-topology-diagnostics) | E2E residual check | Obstruction and interaction diagnostics tutorial |
| Sheaves and cosheaves | Prototype | [`sheaf_consistency_residual`](../reference/api.md#prototype-topology-diagnostics) | E2E residual check | [Layer/head/shard residual gallery](../gallery/sheaf-consistency-residuals.md) |
| Mapper / Reeb / contour topology | Prototype | [`mapper_graph`](../reference/api.md#prototype-topology-diagnostics) | E2E Mapper graph check | [Activation/loss Mapper visual tutorial](../gallery/mapper-reeb-activation-map.md) |
| Morse / Conley / dynamical topology | Prototype | [`dynamical_signature`](../reference/api.md#prototype-topology-diagnostics) | E2E sampled dynamics check | Training-dynamics topology gallery |
| Stratified and singular spaces | Prototype | [`activation_strata`](../reference/api.md#prototype-topology-diagnostics) | E2E activation-strata check | ReLU region and decision-boundary tutorial |
| Nerve / Cech / cover calculus | Prototype | [`metric_cover`, `nerve_graph`](../reference/api.md#prototype-topology-diagnostics) | E2E cover and nerve checks | [Cech-vs-Rips and routing tutorial](../gallery/covers-nerves-routing.md) |
| Knot, braid, and link topology | Prototype | [`braid_crossing_signature`](../reference/api.md#prototype-topology-diagnostics) | E2E braid crossing check | Thread/trajectory entanglement design note |
| Fiber bundles and characteristic classes | Docs | [`TensorBundleSpec`](../reference/api.md#tensorbundle-interoperability) as runtime descriptor | E2E TensorBundle interop check | Gauge/equivariant ML tutorial |
| Topological groups, group actions, quotients | Prototype | [`finite_orbit_signature`, `equivariance_residual`](../reference/api.md#prototype-topology-diagnostics) | E2E finite-orbit and equivariance checks | Equivariance contract utilities |
| Low-dimensional / geometric topology | Prototype | [`mesh_euler_characteristic`](../reference/api.md#prototype-topology-diagnostics) | E2E mesh Euler check | Mesh/scene/handle-surgery explanation |
| Categorical, Grothendieck, and pointless topology | Prototype | [`finite_topology_signature`](../reference/api.md#prototype-topology-diagnostics) | E2E finite open-set lattice check | Sites/locales for observability and privacy |
| Topological vector spaces / function-space topology | Prototype | [`weak_convergence_residual`](../reference/api.md#prototype-topology-diagnostics) | E2E finite-probe weak convergence check | Function-space drift tutorial |

The public Python API exposes the same matrix:

```python
import topoml

for row in topoml.topology_family_coverage_matrix():
    print(row["id"], row["status"], row["api_surface"])
```

Promotion rule: a family moves from docs to prototype only when it has a named
object, a computable invariant, a clear ML/system decision, a null baseline, and
a graph that makes the behavior inspectable.
