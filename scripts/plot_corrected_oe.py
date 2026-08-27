#!/usr/bin/env python3
"""Plot coarse copy-flow corrected O/E from a saved model result."""
import argparse
from pathlib import Path
import sys

import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/dell/a1/microc")
sys.path.insert(0, str(ROOT / "code/cnv_latent_ssm/src"))
from cnv_latent_ssm.graph_expected import aggregate_observed_expected  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cool-uri", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--model-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=10)
    args = parser.parse_args()

    clr = cooler.Cooler(args.cool_uri)
    bins = clr.bins().fetch(args.chrom).reset_index(drop=True)
    observed = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    model = np.load(args.model_npz)
    expected = model["expected"]
    valid = model["valid"].astype(bool)
    _, _, oe, coarse_valid = aggregate_observed_expected(
        observed, expected, valid, args.factor)
    eligible = np.outer(coarse_valid, coarse_valid)
    shown = np.full_like(oe, np.nan)
    shown[eligible] = np.log2(np.maximum(oe[eligible], 2 ** -2))

    chrom_end = bins.end.max() / 1e6
    resolution_kb = int(clr.binsize * args.factor / 1000)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(shown, cmap="RdBu_r", vmin=-2, vmax=2,
                      extent=(0, chrom_end, chrom_end, 0),
                      interpolation="none")
    ax.set(title=f"CN/SV copy-flow corrected O/E ({resolution_kb} kb)",
           xlabel=f"{args.chrom} position (Mb)",
           ylabel=f"{args.chrom} position (Mb)")
    fig.colorbar(image, ax=ax, label="log2[sum(O)/sum(E)], zeros shown at -2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
