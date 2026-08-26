# F1 chr18 copy-flow example

This directory contains the sample-derived intermediate files and outputs used
for the first local evaluation of the additive copy-flow expected-contact
model. Publication of these F1-derived files was explicitly authorized by the
data owner.

## Contents

- `input_tables/`: segment, reference-edge, junction, source/sink, and
  CNV/SV-boundary tables from the source-preserving balance.
- `flow/`: original oriented gGnome peel topology and the capacity-reweighted
  walk-node table used by this evaluation.
- `results/chr18.copy_flow_additive.npz`: O/E, expected components, copy-pair
  pools, external fit, and 500-kb Pearson matrix.
- `results/chr18.copy_flow_additive.png`: O/E and component diagnostic figure.
- `results/F1_chr18.cnv_jcn_balance.png`: sequence-CN and junction-CN summary.

## Important limitation

The source-preserving JCN solution was not completely re-peeled because the
local R environment lacked an installed gGnome package. Instead, the previous
oriented walk topology was capacity-reweighted using the minimum new/old edge
capacity on each walk. Removed flow remains source/sink flow. These files are a
sensitivity result, not a final derivative-allele reconstruction or CALDER
input.

The NPZ arrays are:

```text
oe, expected, cis_expected, external_expected,
cis_copy_pairs, external_copy_pairs, total_copy_pairs,
external_level, external_beta, pearson_500kb, cn, valid
```
