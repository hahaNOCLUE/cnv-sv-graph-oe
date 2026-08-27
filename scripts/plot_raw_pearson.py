#!/usr/bin/env python3
"""Plot 500-kb Pearson from a distance-only O/E matrix."""
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
from cnv_latent_ssm.features import compute_interaction_profile_correlation  # noqa: E402
from cnv_latent_ssm.graph_expected import aggregate_observed_expected  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cool-uri", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--model-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clr = cooler.Cooler(args.cool_uri)
    raw = np.asarray(clr.matrix(balance=False).fetch(args.chrom), float)
    data = np.load(args.model_npz)
    valid = data["valid"].astype(bool)
    decay = data["baseline_decay"]
    index = np.arange(len(raw))
    distance = np.abs(index[:, None] - index[None, :])
    expected = decay[np.minimum(distance, len(decay) - 1)]
    _, _, coarse_oe, coarse_valid = aggregate_observed_expected(
        raw, expected, valid, factor=10)
    pearson = compute_interaction_profile_correlation(
        coarse_oe, coarse_valid, log_transform=True)

    chrom_end = clr.bins().fetch(args.chrom).end.max() / 1e6
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(pearson, cmap="RdBu_r", vmin=-1, vmax=1,
                      extent=(0, chrom_end, chrom_end, 0),
                      interpolation="none")
    ax.set(title="Distance-only O/E: standard 500-kb Pearson",
           xlabel=f"{args.chrom} position (Mb)",
           ylabel=f"{args.chrom} position (Mb)")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)


if __name__ == "__main__":
    main()
