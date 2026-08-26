# F1 chr18 copy-flow example

This directory contains the sample-derived intermediate files and outputs used
for the first local evaluation of the additive copy-flow expected-contact
model. Publication of these F1-derived files was explicitly authorized by the
data owner.

## Contents

- `input_tables/`: segment, reference-edge, junction, source/sink, and
  CNV/SV-boundary tables from the source-preserving balance.
- `flow/`: oriented gGnome peel walks reconstructed directly from the
  source-preserving balanced graph, including node, edge, and walk summaries.
- `results/chr18.copy_flow_additive.npz`: O/E, expected components, copy-pair
  pools, external fit, and 500-kb Pearson matrix.
- `results/chr18.copy_flow_additive.png`: O/E and component diagnostic figure.
- `results/F1_chr18.cnv_jcn_balance.png`: sequence-CN and junction-CN summary.

## Important limitation

The source-preserving graph was newly decomposed with `gGnome::peel`: 66
complete walks were exported, including four circular walks. A final 0.67 CN
of open/source flow could not be peeled into a complete derivative walk and
was deliberately left as source residual rather than absorbed into an SV.
The external-contact model is still a global power law and fitted at the lower
beta bound in this example, so this remains a diagnostic result rather than a
recommended CALDER input.

The additive model uses pair-pool-consistent units:

```text
cis molecule multiplicity: M / P
total copy-pair pool:       CN_i CN_j / P^2
external copy-pair pool:    (CN_i CN_j - M) / P^2
```

The NPZ arrays are:

```text
oe, expected, cis_expected, external_expected,
cis_copy_pairs, external_copy_pairs, total_copy_pairs,
external_level, external_beta, pearson_500kb, cn, valid
```
