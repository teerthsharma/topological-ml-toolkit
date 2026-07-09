# Finite Topology, Dynamics, Braids, And Meshes

This page shows the finite diagnostics that move several topology families from
docs-only language into executable API checks.

```mermaid
flowchart LR
  A["Finite open sets"] --> B["finite_topology_signature"]
  C["Loss or metric curve"] --> D["dynamical_signature"]
  E["Planar sampled strands"] --> F["braid_crossing_signature"]
  G["Vertices, edges, faces"] --> H["mesh_euler_characteristic"]
  B --> I["Topology family coverage matrix"]
  D --> I
  F --> I
  H --> I
```

## Active API

```python
finite = topoml.finite_topology_signature({"a", "b"}, [set(), {"a"}, {"a", "b"}])
dynamics = topoml.dynamical_signature([3.0, 2.0, 1.2, 1.6, 1.1])
braid = topoml.braid_crossing_signature(strands)
mesh = topoml.mesh_euler_characteristic(vertices, edges, faces)
```

## How To Read It

```mermaid
graph TB
  T0["T0 finite space"] --> Sep["Separated enough to distinguish points"]
  T1["T1 finite space"] --> StrongSep["Each point can be isolated from the other"]
  Dyn["Critical indices"] --> Train["Training phase changes"]
  Braid["sigma_i word"] --> Trace["Trajectory interleaving"]
  Chi["Euler characteristic"] --> Shape["Mesh or surface summary"]
```

## Claim Boundary

These APIs are finite sampled diagnostics. They do not prove arbitrary
topological theorems, compute knot polynomials, solve full Conley index
decompositions, or classify manifolds. They give data scientists and systems
engineers inspectable invariants that can be put into tests and benchmark
artifacts.
