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

