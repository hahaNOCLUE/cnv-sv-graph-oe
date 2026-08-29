# F1 chrX refined fold-back example

This snapshot uses the 50-kb continuous CNV segmentation, refined EagleC2
breakpoints (100-kb CNV-boundary tolerance), confidence-prioritized one-to-one
matching, and explicit fold-back handling. The low-confidence call SV26 is
deduplicated against the high-confidence `--` fold-back SV08. Fold-back source
penalty 10 was used for the selected balance and gGnome peel.

The `results` directory also contains the current graph-aware visibility fit:
`chrX.graph_visibility.tsv`, its convergence/holdout QC, and
`chrX.graph_visibility.oe_and_pearson.png`. This fit uses a deterministic 80/20
pixel split and requires at least 90% of expected exposure to come from
same-molecule copy instances at 0.5--5 Mb contour distance. In this snapshot,
holdout row-ratio CV improves from 0.337 with `q=1` to 0.230 with fitted `q`;
`corr(log(q), CN)=0.006`. Reciprocal unknown continuation is represented by
edge-specific mass constrained to `0 <= S_recip <= JCN`, so an unused SV
candidate cannot open free source flow.

The external-decay directory contains the pooled copy-neutral autosome decay
estimate used for follow-up comparison.

Large chromosome-scale observed/expected matrices and RDS cache files are not
included. They can be regenerated from the scripts and the tables committed
here.
