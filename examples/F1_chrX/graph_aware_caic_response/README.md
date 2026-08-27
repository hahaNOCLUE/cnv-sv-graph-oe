# chrX graph-aware CAIC response scan

This directory contains a chromosome-level proof of concept in which the
physical copy-flow expected is frozen and two CN-state-only effective-yield
functions are fitted:

```text
E = g_cis(CN_i, CN_j) * E_cis + g_ext(CN_i, CN_j) * E_external
```

Each response is symmetric and separable,
`g(r_i,r_j)=exp[f(log r_i)+f(log r_j)]`, with five monotone spline knots,
`g(1)=1`, smoothness regularization, and shrinkage toward one. Fits use pooled
CN states rather than segment identity. Cis and external responses are trained
separately on high-confidence component-dominated pixels and include zero
observations.

`chrX.cn_response_sensitivity.tsv` reports the full shrinkage scan. The weakest
fit (`ridge=0.01`) changes `(R_all, R_cis, R_ext)` from approximately
`(2.01, 2.18, 1.33)` to `(1.22, 1.22, 1.17)`, but requires response ranges
`g_cis=1.00-2.28` and `g_ext=0.74-1.54`. It also leaves strong CN-stratified
miscalibration: CN-product quartile sum-ratios are approximately
`2.27, 1.98, 1.00, 0.96`.

The scan therefore demonstrates that a monotone CN response can absorb part of
the global residual but is not accepted as a final correction. The NPZ files
are retained for audit and sensitivity analysis; parameter selection must not
be based on Pearson appearance.

`chrX.graph_aware_caic.ridge_0.01.oe_and_standard_pearson.500kb.png`
shows the 500-kb O/E and its standard unmasked Pearson side by side. Result
figures use this paired layout so the Pearson input is always visible.
