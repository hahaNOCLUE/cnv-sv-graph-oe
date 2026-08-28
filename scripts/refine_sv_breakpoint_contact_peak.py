#!/usr/bin/env python3
"""Refine a paired SV breakpoint by a multiresolution local contact peak."""

from __future__ import annotations

import argparse
from pathlib import Path

import cooler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mcool", type=Path, required=True)
    p.add_argument("--chrom", default="chrX")
    p.add_argument("--left", type=int, required=True)
    p.add_argument("--right", type=int, required=True)
    p.add_argument("--window", type=int, default=1_000_000)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    panels = []
    for resolution in (5_000, 10_000, 25_000):
        clr = cooler.Cooler(f"{a.mcool}::/resolutions/{resolution}")
        left_region = f"{a.chrom}:{a.left-a.window}-{a.left+a.window}"
        right_region = f"{a.chrom}:{a.right-a.window}-{a.right+a.window}"
        matrix = np.asarray(
            clr.matrix(balance=False).fetch(left_region, right_region), dtype=float
        )
        # Smooth on a constant 25-kb physical scale. This stabilizes sparse
        # 5-kb Micro-C pixels without reverting to the original 50-kb bins.
        sigma = max(1.0, 25_000 / resolution)
        smooth = gaussian_filter(np.nan_to_num(matrix), sigma=sigma)
        # Remove separable row/column coverage to localize a focal corner peak.
        row = gaussian_filter(smooth.mean(axis=1), sigma=sigma)
        col = gaussian_filter(smooth.mean(axis=0), sigma=sigma)
        expected = np.outer(row, col) / max(float(smooth.mean()), 1e-12)
        enrichment = np.divide(
            smooth, expected, out=np.zeros_like(smooth), where=expected > 0
        )
        # Local enrichment alone can select a single sparse pixel with tiny
        # expected value. Breakpoint localization therefore uses raw support
        # summed over a fixed 50-kb physical window; enrichment is reported at
        # that independently selected position.
        support_bins = max(1, int(round(50_000 / resolution)))
        support = uniform_filter(matrix, size=support_bins, mode="constant") * support_bins**2
        # Ignore 100 kb at window edges, where convolution support is partial.
        edge = int(np.ceil(100_000 / resolution))
        search = support.copy()
        search[:edge] = search[-edge:] = 0
        search[:, :edge] = search[:, -edge:] = 0
        i, j = np.unravel_index(np.argmax(search), search.shape)
        left_pos = a.left - a.window + (i + 0.5) * resolution
        right_pos = a.right - a.window + (j + 0.5) * resolution
        rows.append(
            {
                "resolution": resolution,
                "refined_left": int(left_pos),
                "refined_right": int(right_pos),
                "left_shift": int(left_pos - a.left),
                "right_shift": int(right_pos - a.right),
                "local_enrichment": float(enrichment[i, j]),
                "window_50kb_contact_support": float(search[i, j]),
                "smoothed_count": float(smooth[i, j]),
            }
        )
        panels.append((resolution, enrichment, left_region, right_region))

    result = pd.DataFrame(rows)
    result.to_csv(a.output_dir / "SV07.multiresolution_breakpoint_refinement.tsv",
                  sep="\t", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for axis, (resolution, enrichment, _, _) in zip(axes, panels):
        extent = (
            (a.right-a.window) / 1e6, (a.right+a.window) / 1e6,
            (a.left+a.window) / 1e6, (a.left-a.window) / 1e6,
        )
        image = axis.imshow(enrichment, cmap="magma", vmin=0,
                            vmax=np.nanpercentile(enrichment, 99.5),
                            extent=extent, interpolation="none")
        axis.axhline(a.left / 1e6, color="#00ffff", ls="--", lw=1)
        axis.axvline(a.right / 1e6, color="#00ffff", ls="--", lw=1)
        hit = result[result.resolution.eq(resolution)].iloc[0]
        axis.scatter(hit.refined_right / 1e6, hit.refined_left / 1e6,
                     marker="x", s=60, color="white")
        axis.set(title=f"{resolution//1000}-kb local enrichment",
                 xlabel="right breakpoint (Mb)", ylabel="left breakpoint (Mb)")
        fig.colorbar(image, ax=axis, shrink=.75)
    fig.savefig(a.output_dir / "SV07.multiresolution_breakpoint_refinement.png",
                dpi=220)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
