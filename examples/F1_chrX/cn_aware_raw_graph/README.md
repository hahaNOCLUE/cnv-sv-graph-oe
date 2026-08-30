# F1 chrX CN-aware raw balancing followed by graph O/E

This benchmark uses one fixed sequence:

1. fit a multiplicative raw-contact bias so corrected cis marginals are
   proportional to continuous copy number;
2. divide the original raw counts by the fixed physical genome-graph expected
   multiplied by that bias and a global scale;
3. compute 500-kb Pearson separately within the hg38 chrX p and q arms;
4. run CALDER at the native 50-kb O/E resolution separately by arm.

The balancing fit used `0.5 <= bias <= 2`, damping 0.5, and excluded the hg38
centromere interval 58.1--63.8 Mb. The compressed arm-specific corrected-O/E
tables are included so CALDER can be reproduced without the 92-MB dense NPZ.

`comparison_metrics.tsv` compares calls against the CALDER hg38 reference.
The CN-aware baseline improves both arms relative to graph-only `q=1` and is
slightly better than the graph-distance visibility fit, but q-arm Pearson and
A/B agreement remain below the ICE-balanced benchmark.
