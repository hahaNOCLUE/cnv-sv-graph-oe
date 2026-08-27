#!/usr/bin/env python3
"""Compare distance-only and copy-flow corrected O/E on one chromosome."""
import argparse
from pathlib import Path

import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cool-uri", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--corrected-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clr = cooler.Cooler(args.cool_uri)
    raw = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    data = np.load(args.corrected_npz)
    valid = data["valid"].astype(bool)
    baseline = data["baseline_decay"]
    corrected = data["oe"]
    distance = np.abs(np.arange(len(raw))[:, None] - np.arange(len(raw))[None, :])
    raw_expected = baseline[np.minimum(distance, len(baseline) - 1)]
    raw_oe = np.divide(raw, raw_expected, out=np.zeros_like(raw),
                       where=raw_expected > 0)
    eligible = np.outer(valid, valid)

    def display_log2(oe):
        shown = np.full_like(oe, np.nan, dtype=float)
        shown[eligible] = np.log2(np.maximum(oe[eligible], 2 ** -2))
        return shown

    extent = (0, clr.bins().fetch(args.chrom).end.max() / 1e6,
              clr.bins().fetch(args.chrom).end.max() / 1e6, 0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    titles = ["Distance-only O/E (uncorrected)",
              "CN/SV copy-flow O/E (corrected)"]
    for ax, matrix, title in zip(axes, (raw_oe, corrected), titles):
        image = ax.imshow(display_log2(matrix), cmap="RdBu_r", vmin=-2, vmax=2,
                          extent=extent, interpolation="none", rasterized=True)
        ax.set(title=title, xlabel=f"{args.chrom} position (Mb)",
               ylabel=f"{args.chrom} position (Mb)")
    cbar = fig.colorbar(image, ax=axes, shrink=.82)
    cbar.set_label("log2(O/E), zeros shown at -2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
