# F1 chrX refined fold-back example

This snapshot uses the 50-kb continuous CNV segmentation, refined EagleC2
breakpoints (100-kb CNV-boundary tolerance), confidence-prioritized one-to-one
matching, and explicit fold-back handling. The low-confidence call SV26 is
deduplicated against the high-confidence `--` fold-back SV08. Fold-back source
penalty 10 was used for the selected balance and gGnome peel.

The `results` directory contains the CNV-bin/CBS/SV overview, the balance plot,
the graph-aware CN-response O/E with standard unmasked 500-kb Pearson, and the
local raw contact map for unmatched SV14. The external-decay directory contains
the pooled copy-neutral autosome decay estimate used for follow-up comparison.

Large chromosome-scale observed/expected matrices and RDS cache files are not
included. They can be regenerated from the scripts and the tables committed
here.
