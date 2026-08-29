#!/usr/bin/env python3
"""Plot graph-aware O/E and Pearson with a highlighted genomic interval."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True, help="Interval start in Mb")
    parser.add_argument("--end", type=float, required=True, help="Interval end in Mb")
    parser.add_argument("--chrom", default="chrX")
    args = parser.parse_args()

    data = np.load(args.npz)
    oe = data["oe"]
    pearson = data["pearson_500kb"]
    chromosome_end = oe.shape[0] * 0.05
    extent = (0, chromosome_end, chromosome_end, 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(
        np.log2(np.where(oe > 0, oe, np.nan)),
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        extent=extent,
        interpolation="none",
    )
    axes[0].set_title("graph-aware log2(O/E)")
    axes[1].imshow(
        pearson,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=extent,
        interpolation="none",
    )
    axes[1].set_title("standard 500-kb Pearson")

    label = f"{args.start:.3f}–{args.end:.3f} Mb"
    for ax in axes:
        ax.axvspan(args.start, args.end, color="#00c8ff", alpha=0.10, zorder=3)
        ax.axhspan(args.start, args.end, color="#00c8ff", alpha=0.10, zorder=3)
        for position in (args.start, args.end):
            ax.axvline(position, color="#007c91", lw=1.2, ls="--", zorder=4)
            ax.axhline(position, color="#007c91", lw=1.2, ls="--", zorder=4)
        ax.add_patch(
            plt.Rectangle(
                (args.start, args.start),
                args.end - args.start,
                args.end - args.start,
                fill=False,
                edgecolor="#00c8ff",
                linewidth=2.0,
                zorder=5,
            )
        )
        ax.text(
            (args.start + args.end) / 2,
            1.5,
            label,
            ha="center",
            va="top",
            fontsize=9,
            color="#006779",
            bbox={"facecolor": "white", "edgecolor": "#00c8ff", "alpha": 0.85, "pad": 2},
            zorder=6,
        )
        ax.set_xlabel(f"{args.chrom} position (Mb)")
        ax.set_ylabel(f"{args.chrom} position (Mb)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor="white")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
