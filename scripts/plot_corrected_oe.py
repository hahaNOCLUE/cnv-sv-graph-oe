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
from cnv_latent_ssm.features import compute_interaction_profile_correlation  # noqa: E402


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
    pearson = compute_interaction_profile_correlation(
        oe, coarse_valid, log_transform=True)
    shown = np.full_like(oe, np.nan)
    shown[eligible] = np.log2(np.maximum(oe[eligible], 2 ** -2))

    chrom_end = bins.end.max() / 1e6
    resolution_kb = int(clr.binsize * args.factor / 1000)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    image = axes[0].imshow(shown, cmap="RdBu_r", vmin=-2, vmax=2,
                           extent=(0, chrom_end, chrom_end, 0),
                           interpolation="none")
    axes[0].set(title=f"Corrected O/E ({resolution_kb} kb)",
                xlabel=f"{args.chrom} position (Mb)",
                ylabel=f"{args.chrom} position (Mb)")
    correlation = axes[1].imshow(
        pearson, cmap="RdBu_r", vmin=-1, vmax=1,
        extent=(0, chrom_end, chrom_end, 0), interpolation="none")
    axes[1].set(title=f"Standard Pearson ({resolution_kb} kb, unmasked)",
                xlabel=f"{args.chrom} position (Mb)",
                ylabel=f"{args.chrom} position (Mb)")
    fig.colorbar(image, ax=axes[0], shrink=.82,
                 label="log2[sum(O)/sum(E)], zeros shown at -2")
    fig.colorbar(correlation, ax=axes[1], shrink=.82,
                 label="Pearson correlation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
