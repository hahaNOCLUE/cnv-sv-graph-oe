# F1 chrX convex Poisson flow snapshot

This directory is one internally consistent graph run. All balance tables,
peeled walks, and displayed O/E results were generated from the same balance
solution.

The junction model is fixed to

```text
Y_e ~ Poisson(a * J_e + b_e)
```

with one shared `a = 700.765774417`, local count background `b_e`, hard
side-flow equality, non-negative REF/JCN/source flow, and an L1 penalty of
1000 per unit of allowed endpoint source. EagleC2 probabilities select the
candidate set only. CN-derived JCN targets, probability penalties, reciprocal
continuation variables, and soft flow penalties are disabled.

Source is allowed only at CN discontinuities and chromosome telomeres. The
CVXPY CLARABEL solution is `optimal`; the maximum hard-flow residual is
`2.90e-7` CN. Ten of 21 candidate junctions have JCN greater than `1e-6`.

The gGnome peel used `embed.loops = TRUE`. Its cache is keyed by the MD5
fingerprint in `input_tables/F1_chr18.gGnome_peel.graph_fingerprint.txt`; the
matching input hashes are recorded in `F1_chr18.gGnome_run_manifest.tsv`.
The peel contains 50 exported walks. Physical expected excludes complete
walks below 0.1 CN because two 0.01-CN residual walks contain pathological
loop expansions of 1,444 and 397 nodes.

The O/E result uses the fixed pooled-autosome decay, `q_i = 1`, no CN-response,
and no CAIC/ICE layer. A single global library scale `s = 1.7389094324` is
retained. Diagnostics including zeros are:

```text
all pixels sum(O)/sum(E)            0.5071
cis-dominated pixels sum(O)/sum(E)  0.4977
ext-dominated pixels sum(O)/sum(E)  0.6858
```

These values are reported without post-hoc correction; this deliberately
shows that the simplified physical expected currently overestimates contact.

The primary 500-kb Pearson analysis is chromosome-arm specific. chrX p arm
(`<58.1 Mb`) and q arm (`>=63.8 Mb`) interaction profiles are correlated
separately. The 58.1--63.8 Mb centromere interval and every p-by-q correlation
are stored and plotted as missing values rather than contributing to either
arm.
