#!/usr/bin/env python3
"""CN-aware balance raw contacts, then divide by a fixed graph expected."""
import argparse
from pathlib import Path
import sys

import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/dell/a1/microc")
sys.path.insert(0, str(ROOT / "code/cnv_latent_ssm/src"))
from cnv_latent_ssm.cn_aware_balance import fit_cn_aware_balance
from cnv_latent_ssm.graph_expected import aggregate_observed_expected
from cnv_latent_ssm.features import compute_interaction_profile_correlation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cool-uri", required=True)
    p.add_argument("--chrom", default="X")
    p.add_argument("--graph-model", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--centromere-start", type=int, default=58_100_000)
    p.add_argument("--centromere-end", type=int, default=63_800_000)
    p.add_argument("--damping", type=float, default=.5)
    p.add_argument("--bias-min", type=float, default=.5)
    p.add_argument("--bias-max", type=float, default=2.)
    p.add_argument("--tolerance", type=float, default=.01)
    p.add_argument("--max-iterations", type=int, default=200)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(args.cool_uri)
    raw = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    bins = clr.bins().fetch(args.chrom).reset_index(drop=True)
    model = np.load(args.graph_model)
    graph_expected = np.asarray(model["graph_expected"], float)
    cn = np.asarray(model["cn"], float)
    valid = np.asarray(model["valid"], bool)
    centers = (bins.start.to_numpy() + bins.end.to_numpy()) / 2
    valid &= ~((centers >= args.centromere_start) & (centers < args.centromere_end))
    mask = np.outer(valid, valid) & np.isfinite(raw)
    np.fill_diagonal(mask, False)

    fit = fit_cn_aware_balance(
        raw, cn, valid, mask, damping=args.damping,
        bias_min=args.bias_min, bias_max=args.bias_max,
        tolerance=args.tolerance, max_iterations=args.max_iterations)
    supported = mask.any(axis=1) & (np.where(mask, raw, 0.).sum(axis=1) > 0)
    valid &= supported
    mask &= np.outer(valid, valid)
    corrected_raw = np.divide(raw, np.outer(fit.bias, fit.bias),
                              out=np.zeros_like(raw), where=mask)
    scale = float(corrected_raw[mask].sum() / graph_expected[mask].sum())
    expected = scale * np.outer(fit.bias, fit.bias) * graph_expected
    usable = np.outer(valid, valid) & (expected > 0)
    oe = np.divide(raw, expected, out=np.zeros_like(raw), where=usable)

    _, _, oe500, valid500 = aggregate_observed_expected(raw, expected, valid, 10)
    n = len(valid500)
    centers500 = (np.arange(n) + .5) * 500_000
    arms = np.full(n, "centromere", object)
    arms[centers500 < args.centromere_start] = "p"
    arms[centers500 >= args.centromere_end] = "q"
    pearson = np.full((n, n), np.nan)
    for arm in ("p", "q"):
        ix = np.flatnonzero(arms == arm)
        pearson[np.ix_(ix, ix)] = compute_interaction_profile_correlation(
            oe500[np.ix_(ix, ix)], valid500[ix], log_transform=True)

    stem = args.chrom.replace("chr", "")
    np.savez_compressed(
        args.output_dir / f"{stem}.cn_aware_raw_graph_oe.npz",
        oe=oe, expected=expected, graph_expected=graph_expected,
        corrected_raw=corrected_raw, cn=cn, valid=valid,
        capture_visibility=fit.bias, global_scale=scale,
        oe_500kb=oe500, pearson_500kb=pearson, pearson_arms=arms,
        centromere_start_bp=args.centromere_start,
        centromere_end_bp=args.centromere_end)
    pd.DataFrame({"chrom": bins.chrom, "start": bins.start, "end": bins.end,
                  "cn": cn, "bias": fit.bias, "valid": valid}).to_csv(
        args.output_dir / f"{stem}.cn_aware_bias.tsv", sep="\t", index=False)
    pd.DataFrame(fit.history, columns=["iteration", "target_scale",
                                      "max_abs_log_ratio"]).to_csv(
        args.output_dir / f"{stem}.cn_aware_history.tsv", sep="\t", index=False)

    extent=(0, bins.end.max()/1e6, bins.end.max()/1e6, 0)
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    ax[0].imshow(np.log2(np.where(oe > 0, oe, np.nan)), cmap="RdBu_r",
                 vmin=-2, vmax=2, extent=extent, interpolation="none")
    ax[0].set_title("CN-aware raw correction + graph log2(O/E)")
    ax[1].imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1,
                 extent=extent, interpolation="none")
    ax[1].set_title("500-kb Pearson (p/q arms separately)")
    for a in ax:
        a.set(xlabel=f"{args.chrom} position (Mb)", ylabel=f"{args.chrom} position (Mb)")
    fig.savefig(args.output_dir / f"{stem}.cn_aware_raw_graph_oe_and_pearson.png", dpi=220)
    print(args.output_dir)
    print("iterations", fit.iterations, "converged", fit.converged,
          "max_error", fit.max_abs_log_ratio, "scale", scale)


if __name__ == "__main__":
    main()
