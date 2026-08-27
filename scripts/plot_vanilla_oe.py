#!/usr/bin/env python3
"""Plot a vanilla chromosome-wide distance O/E benchmark."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=10)
    args = parser.parse_args()

    clr = cooler.Cooler(args.cool_uri)
    bins = clr.bins().fetch(args.chrom).reset_index(drop=True)
    observed = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    weight = np.asarray(bins["weight"], float)
    valid = np.isfinite(weight) & (weight > 0)

    factor = args.factor
    n = int(np.ceil(len(valid) / factor))
    size = n * factor
    valid_pairs = np.outer(valid, valid)
    padded = np.zeros((size, size), float)
    padded[:len(valid), :len(valid)] = np.where(valid_pairs, observed, 0)
    coarse_observed = padded.reshape(n, factor, n, factor).sum(axis=(1, 3))
    coarse_valid = np.array([
        valid[i * factor:min((i + 1) * factor, len(valid))].any()
        for i in range(n)
    ])
    coarse_pairs = np.outer(coarse_valid, coarse_valid)

    expected_by_distance = np.full(n, np.nan)
    for distance in range(n):
        values = np.diag(coarse_observed, k=distance)
        eligible = np.diag(coarse_pairs, k=distance) & np.isfinite(values)
        if eligible.any():
            expected_by_distance[distance] = values[eligible].mean()
    distance = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    expected = expected_by_distance[distance]
    oe = np.divide(coarse_observed, expected, out=np.zeros_like(coarse_observed),
                   where=np.isfinite(expected) & (expected > 0))
    shown = np.full_like(oe, np.nan)
    shown[coarse_pairs] = np.log2(np.maximum(oe[coarse_pairs], 2 ** -2))

    chrom_end = bins.end.max() / 1e6
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(shown, cmap="RdBu_r", vmin=-2, vmax=2,
                      extent=(0, chrom_end, chrom_end, 0),
                      interpolation="none")
    ax.set(title="Vanilla chromosome-wide distance O/E (500 kb)",
           xlabel=f"{args.chrom} position (Mb)",
           ylabel=f"{args.chrom} position (Mb)")
    fig.colorbar(image, ax=ax, label="log2(O/E), zeros shown at -2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
