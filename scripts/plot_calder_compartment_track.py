#!/usr/bin/env python3
"""Plot chromosome-wide CALDER compartment tracks at a fixed bin size."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


def mode(series):
    values = series.dropna().astype(str)
    return values.value_counts().index[0] if len(values) else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chrom", default="chrX")
    p.add_argument("--bin-size", type=int, default=50000)
    p.add_argument("--chrom-end", type=int, required=True)
    p.add_argument("--centromere-start", type=int)
    p.add_argument("--centromere-end", type=int)
    args = p.parse_args()

    data = pd.read_csv(args.input, sep="\t")
    data = data[data["chr"].eq(args.chrom)].copy()
    data["bin"] = ((data.pos_start - 1) // args.bin_size).astype(int)
    data["ab"] = data.comp_name.str.split(".", regex=False).str[0]
    data["sub"] = data.comp_name.str.split(".", regex=False).str[:2].str.join(".")
    track = data.groupby("bin", as_index=False).agg(
        continuous=("continous_rank", "mean"), ab=("ab", mode), sub=("sub", mode))
    n = int(np.ceil(args.chrom_end / args.bin_size))
    bins = pd.DataFrame({"bin": np.arange(n)})
    track = bins.merge(track, how="left", on="bin")
    track["signed_continuous"] = track.continuous * track.ab.map({"A": 1, "B": -1})
    x = (track.bin + .5) * args.bin_size / 1e6

    labels = ["B.2", "B.1", "A.2", "A.1"]
    colors = ["#204b9b", "#79a9dc", "#f6a36b", "#c51b32"]
    code = {label: i for i, label in enumerate(labels)}
    subcode = track["sub"].map(code).to_numpy(float)
    abcode = track.ab.map({"B": 0, "A": 1}).to_numpy(float)

    fig, axes = plt.subplots(3, 1, figsize=(16, 6.8), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1, 1]},
                             constrained_layout=True)
    axes[0].axhline(0, color="0.45", lw=.8)
    axes[0].plot(x, track.signed_continuous, color="#222222", lw=.8)
    axes[0].fill_between(x, 0, track.signed_continuous,
                         where=track.signed_continuous >= 0, color="#d73027", alpha=.7)
    axes[0].fill_between(x, 0, track.signed_continuous,
                         where=track.signed_continuous < 0, color="#2166ac", alpha=.7)
    axes[0].set_ylabel("CALDER signed\ncontinuous rank")
    axes[0].set_title(f"F1 {args.chrom}: corrected O/E CALDER compartments ({args.bin_size//1000} kb)")

    axes[1].imshow(abcode[None, :], aspect="auto", interpolation="none",
                   cmap=ListedColormap(["#2166ac", "#d73027"]), vmin=0, vmax=1,
                   extent=(0, args.chrom_end/1e6, 0, 1))
    axes[1].set_yticks([.5], ["A/B"])
    axes[2].imshow(subcode[None, :], aspect="auto", interpolation="none",
                   cmap=ListedColormap(colors), vmin=0, vmax=3,
                   extent=(0, args.chrom_end/1e6, 0, 1))
    axes[2].set_yticks([.5], ["subcompartment"])
    axes[2].set_xlabel(f"{args.chrom} position (Mb)")
    for ax in axes:
        ax.set_xlim(0, args.chrom_end/1e6)
        if args.centromere_start is not None and args.centromere_end is not None:
            ax.axvspan(args.centromere_start/1e6, args.centromere_end/1e6,
                       color="0.55", alpha=.18, lw=0)
    handles = [plt.Line2D([0], [0], color=c, lw=7, label=l)
               for l, c in zip(labels, colors)]
    axes[2].legend(handles=handles, ncol=4, loc="upper center",
                   bbox_to_anchor=(.5, -.65), frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    track.to_csv(args.output.with_suffix(".50kb.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
