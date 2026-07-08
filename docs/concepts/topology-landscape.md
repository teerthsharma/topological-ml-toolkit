# Topology Landscape

The current toolkit starts with topological data analysis: point clouds,
Vietoris-Rips filtrations, Betti numbers, persistence diagrams, and time-delay
embeddings. That is useful, but it is only one part of topology.

This page is the expansion map. It records the major topology families the
toolkit should teach, prototype, or eventually support as stable APIs.

## What The Current Stack Covers

The current code and documentation are strongest in:

- persistent homology over bounded Vietoris-Rips and witness complexes;
- Betti numbers for connected components, loops, and voids;
- point-cloud and manifold embeddings;
- spherical and hyperbolic locality;
- Voronoi-style routing and memory layout;
- spectral contraction and Lyapunov-style stability language;
- topology-guided scheduling and routing as roadmap material.

That is a strong base, but it overrepresents "holes in point clouds" and
underrepresents topology as a full language of continuity, covers, maps,
symmetry, gluing, dynamics, and computation.

## Broad Topology Taxonomy

Wikipedia's overview frames topology as the study of properties preserved under
continuous deformation, such as connectedness, compactness, and dimension, and
lists major subfields including general topology, algebraic topology,
differential topology, geometric topology, and generalized/categorical
directions. This project uses that as a broad scaffold, then translates each
family into ML and systems terms.

## Missing Or Underused Families

| Family | Object / invariant | Why ML or systems should care | Track |
| --- | --- | --- | --- |
| Point-set / general topology | Open sets, neighborhoods, continuity, compactness, separation, connectedness | Makes "topological" mean more than holes. Gives precise language for convergence, observability, finite covers, data validity, and continuity of preprocessing pipelines. | Docs-only first |
| Metric, uniform, and proximity spaces | Entourages, Cauchy behavior, total boundedness, nearness without one fixed metric | Useful for embedding APIs, approximate nearest-neighbor search, drift detection, cache locality, and tolerance contracts across backends. | Active prototype for metric covers |
| Domain theory / Scott topology | Posets, directed suprema, monotone maps, fixed points, computable continuity | Natural fit for compiler semantics, schedulers, streaming dataflow, abstract interpretation, and monotone runtime proofs. | Core later |
| Homotopy theory | Fundamental group, higher homotopy groups, path classes, deformation classes | Distinguishes optimization paths and system-state trajectories that homology can collapse into the same count. Useful for "safe deformation" of plans, policies, or schedules. | Prototype |
| Cohomology beyond Betti counts | Cocyles, cup products, obstruction classes | Encodes interactions and global consistency, not just feature counts. Candidate for feature interaction diagnostics and distributed consistency checks. | Prototype |
| Sheaves and cosheaves | Local data glued by restriction maps; sheaf cohomology residuals | Direct model for layers, heads, shards, feature stores, sensor networks, and telemetry systems where local views must agree. | Active prototype for residual diagnostics |
| Mapper / Reeb / contour topology | Level-set graphs of scalar filters; critical changes in shape | Gives interpretable maps of datasets, activations, loss surfaces, entropy filters, and routing states. More approachable to practitioners than raw chain complexes. | Active prototype for Mapper graphs |
| Morse / Conley / dynamical topology | Critical points, gradient flows, invariant sets, attractor decompositions | Turns training dynamics, online learning regimes, and optimizer stability into objects that can be graphed and falsified. | Prototype |
| Stratified and singular spaces | Spaces with strata, boundaries, corners, singularities | Models ReLU regions, decision boundaries, failure surfaces, discontinuities, and non-manifold data instead of pretending every dataset is smooth. | Prototype |
| Nerve / Cech / cover calculus | Covers and nerves; topology-preserving summaries | Generalizes Vietoris-Rips into explicit cover-based batching, cache regions, privacy-preserving summaries, and routing cells. | Core later |
| Knot, braid, and link topology | Embedded curves, crossings, linking and braid invariants | Candidate for thread interleavings, trajectory entanglement, attention-head coupling, robotics traces, and multi-agent coordination. | Docs-only then prototype |
| Fiber bundles and characteristic classes | Fibers over base spaces, sections, holonomy, Chern/Stiefel-Whitney-style obstructions | Language for gauge/equivariant ML, layerwise parameter transport, distributed model alignment, and symmetry defects. | Docs-only first |
| Topological groups, group actions, quotients | Orbits, stabilizers, quotient spaces | Makes symmetry-aware model tests and equivariance contracts explicit; formalizes SO(3), rotations, and quotient embeddings. | Prototype |
| Low-dimensional / geometric topology | Orientability, genus, handles, surgery, 3/4-manifold decompositions | Good for mesh/scene systems, memory-layout surgery, graph rewrites, and explaining where "topological surgery" claims need real handle calculus. | Docs-only first |
| Categorical, Grothendieck, and pointless topology | Sites, locales, topoi, open-set lattices instead of point sets | Gives compositional foundations for observability, privacy, distributed knowledge, and sheaf-ready APIs. | Docs-only first |
| Topological vector spaces / function-space topology | Weak/strong topologies, convergence of functions and distributions | Relevant to model convergence, kernel methods, infinite-width limits, optimizer stability, and distribution shift. | Docs-only then prototype |

## Implementation Priority

The library should not try to implement all of topology at once. That would
produce a broad but useless catalog. The right path is:

1. **Document the full landscape.** Make each topology family visible with a
   plain-language explanation and at least one ML/systems use case.
2. **Prototype where a graph helps.** Mapper/Reeb graphs, sheaves, metric covers,
   and dynamical topology should get visual notebooks before hard APIs.
3. **Promote only falsifiable wins to core.** A family becomes core only when it
   has a stable object, invariant, baseline, failure mode, and benchmark gate.

## Candidate Milestones

| Milestone | Topology object | First deliverable | Promotion gate |
| --- | --- | --- | --- |
| Mapper data maps | Cover + nerve graph | Gallery page for activations and tabular data | Beats PCA/UMAP-only diagnostics on interpretability task |
| Sheaf consistency | Local sections over graph | Prototype residual over heads/layers/shards | Detects injected inconsistency better than scalar variance |
| Cech/nerve covers | Explicit covers | Cover-based batching API | Same-budget comparison vs Vietoris-Rips and random covers |
| Dynamical topology | Attractors and recurrent sets | Training trajectory gallery | Separates stable/unstable runs before final metric diverges |
| Symmetry topology | Group actions and quotients | Equivariance test utilities | Catches symmetry violations missed by ordinary unit tests |
| Function-space topology | Weak convergence summaries | Distribution-shift diagnostics | Correlates with held-out degradation better than norm drift |

## Rule For Future Code

Every new topology family must answer five questions before implementation:

1. What is the topological object?
2. What invariant or construction do we compute?
3. What ML or systems decision does it change?
4. What baseline or null model can falsify it?
5. What graph makes the idea understandable to a non-topologist?

If any answer is missing, the family stays in docs or gallery until it earns
code.
