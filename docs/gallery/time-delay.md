# Time Delay

Periodic signals become loops after time-delay embedding.

![Time delay embedding](../assets/time_delay_embedding.svg)

Pipeline:

```mermaid
flowchart LR
  A["1D signal"] --> B["Delay vectors"]
  B --> C["Point cloud"]
  C --> D["Persistent homology"]
  D --> E["Periodicity features"]
```

```mermaid
sequenceDiagram
  participant Signal as Scalar signal
  participant Delay as Delay embedding
  participant PH as Persistent homology
  participant ML as ML feature table
  Signal->>Delay: choose dimension and tau
  Delay->>PH: build point cloud
  PH->>ML: Betti curve or persistence image
  ML-->>Signal: auditable periodicity feature
```

## Claim Boundary

The active API builds delay vectors and topology features. It does not estimate
the best lag automatically, prove periodicity for arbitrary signals, or beat a
spectral baseline without a task benchmark.
