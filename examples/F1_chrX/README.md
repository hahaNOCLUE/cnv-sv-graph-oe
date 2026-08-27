# F1 chrX copy-flow test

This example tests the CN/SV copy-flow model on chrX, which has 17
intra-chromosomal and five inter-chromosomal EagleC2 calls in the genome-wide
50-kb call set.

## Model assumptions

- CNVkit log2 ratios are converted with reference ploidy 2.
- Physical molecule normalization uses chrX ploidy 1.
- Inter-chromosomal calls are represented as explicit external derivative
  edges rather than free source flow.
- Same-molecule decay and the intra-molecular collision floor are fitted
  jointly, keeping the decay positive and monotone.
- The 500-kb Pearson matrix uses complete interaction profiles with no
  reference-distance or graph-distance mask.
- chrX lacks a matched junction-depth calibration in this test. Junction CN is
  therefore driven by CN-step targets and graph flow balance; the result is a
  diagnostic benchmark, not a final calibrated reconstruction.

## Contents

- `input_tables/`: balanced sequence, reference, junction, source, and graph
  connectivity QC tables.
- `flow/`: 41 gGnome peeled walks (three circular; no singleton walks).
- `results/chrX.copy_flow_joint_decay.npz`: expected components, O/E,
  copy-pair pools, 500-kb Pearson, and fitted decay arrays.
- `results/chrX.copy_flow_corrected_oe.500kb.png`: corrected O/E computed as
  `sum(O_50kb) / sum(E_50kb)`.
- `results/chrX.copy_flow_diagnostics.standard_pearson.png`: four-panel graph
  and O/E diagnostic with standard, unmasked Pearson.
- `results/chrX.vanilla_distance_oe.500kb.png`: chromosome-wide distance-only
  O/E benchmark without CN, SV, graph, or ploidy correction.
- `results/chrX.vanilla_distance_oe.standard_500kb_pearson.png`: corresponding
  standard Pearson benchmark.

## Current diagnostics

The joint fit eliminates long-range decay clipping (`same_decay == 0`: 0%).
However, `sum(O)/sum(E)` remains about 2.01 overall, 2.18 for cis-dominated
pixels, and 1.33 for external-dominated pixels. The remaining mismatch is
concentrated in graph-supported same-molecule contacts.
