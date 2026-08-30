# F1 chrX latent-CN / sparse-source snapshot

This snapshot uses one internally consistent chrX graph run. It fixes two
upstream problems present in the earlier example:

1. chrX 58.1--63.8 Mb is an excluded centromere interval. Its 114 50-kb bins
   remain missing (`weight=0`) and CBS segments the p and q arms separately;
   no CN interpolation or graph reference edge crosses the gap.
2. Parent CBS copy number is latent rather than an exact right-hand side.
   Every graph segment cut from the same parent shares one latent CN.

The balance model is

```text
CN observation loss (Laplace)
+ Y_e ~ Poisson(b_e + a * J_e), shared a = 700.765774417
+ exact side-flow conservation
```

EagleC2 probability selects candidate edges only. CN-jump JCN targets and
probability-weighted JCN penalties are disabled. Missing reciprocal
continuation is edge-specific and constrained by `0 <= U_e,k <= J_e`.

Optimization is lexicographic:

1. minimize CN plus junction data likelihood;
2. retain solutions within one NLL unit and eliminate unresolved source
   locations one by one until the support is inclusion-minimal;
3. with those locations fixed, minimize source mass.

The final solution has maximum hard-flow residual `1.43e-7` CN, 36 active
non-telomeric unresolved source locations, total JCN `25.3744`, and total
generic/reciprocal source CN `43.7242/21.6575`. The largest latent-CN shift is
reported rather than hidden: CNSEG007 changes from `1.9015` to `7.6132`, a
16.2-sigma conflict between the NeoLoop CN profile and junction evidence.

The gGnome peel used `embed.loops = TRUE`, graph fingerprint
`74330a66b69f79068be1bc0e89802a3d`, and exported 81 walks (two circular) with
zero residual graph flow. The O/E panel uses pooled-autosome decay,
`min_walk_cn=0.1`, fixed `q_i=1`, and one global scale `s=1.7855017540`.
Pearson is computed separately for the p and q arms; centromere and cross-arm
blocks are missing.

Zero-inclusive diagnostics are:

```text
all pixels sum(O)/sum(E)             0.5947
cis-dominated pixels sum(O)/sum(E)   0.5520
external-dominated sum(O)/sum(E)     1.4398
centromere valid bins                0
```

Large NPZ and RDS objects are intentionally omitted. The committed balance
tables, peel TSVs, fingerprints, QC tables, and rendered CN/JCN and O/E /
Pearson panels provide the reproducible result snapshot.
