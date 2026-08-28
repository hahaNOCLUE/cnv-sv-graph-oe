# F1 chrX genome-component pair-conservation rerun

This snapshot contains the chrX-connected genome graph rerun after two model
corrections:

1. Same-molecule occurrence pairs from duplicated segments are retained rather
   than capped by `min(CN_i, CN_j)`.
2. Source flow is forbidden at ordinary reference-continuous segment sides.
   Only chromosome ends and true CBS CN transitions permit unresolved source.

The component spans 23 chromosomes, 782 segments and 184 SV edges. Flow
residual is below `9e-16`; forbidden source CN is zero. gGnome exported 493
walks. The displayed expected uses pooled copy-neutral autosome decay and no
CN-response layer.

## Sum-ratio QC

| model | R_all | R_cis | R_ext |
|---|---:|---:|---:|
| nominal pair cap | 1.016754 | 0.971891 | 3.479580 |
| uncapped occurrences | 0.911463 | 0.869606 | 3.479580 |

Fractional/subclonal CN marginals do not define joint clone occupancy, so
nominal `CN_i*CN_j/P^2` exceedances are retained as QC rather than clipped.
The remaining dominant mismatch is the external/unknown component.

Large NPZ matrices and RDS caches are omitted. `calder/` contains the existing
50-kb CALDER track produced from the corrected graph+CN-response O/E benchmark;
it is included for reproducibility but is not the uncapped no-CN-response rerun.
