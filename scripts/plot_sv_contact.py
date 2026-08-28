#!/usr/bin/env python3
"""Plot a local contact map and mark an intrachromosomal SV."""
import argparse

import cooler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cool-uri", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--pos1", type=int, required=True)
    p.add_argument("--pos2", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--end", type=int, required=True)
    p.add_argument("--label", default="SV")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    clr = cooler.Cooler(args.cool_uri)
    region = f"{args.chrom}:{args.start}-{args.end}"
    raw = np.asarray(clr.matrix(balance=False).fetch(region), float)
    shown = np.log1p(raw)
    finite = shown[np.isfinite(shown) & (shown > 0)]
    vmax = np.quantile(finite, .995) if finite.size else 1
    extent = (args.start/1e6, args.end/1e6, args.end/1e6, args.start/1e6)
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(shown, cmap="OrRd", vmin=0, vmax=vmax, extent=extent,
                   interpolation="none")
    x, y = args.pos1/1e6, args.pos2/1e6
    ax.axvline(x, color="#1764ab", lw=1.2, ls="--")
    ax.axhline(y, color="#1764ab", lw=1.2, ls="--")
    ax.scatter([x], [y], marker="x", s=130, linewidths=2.4,
               color="#082f6b", zorder=5)
    ax.text(x, y, f"  {args.label}", color="#082f6b", fontsize=11,
            ha="left", va="bottom")
    ax.set(title=f"{args.label}: raw contact map (log1p counts)",
           xlabel=f"{args.chrom} position (Mb)",
           ylabel=f"{args.chrom} position (Mb)")
    fig.colorbar(im, ax=ax, shrink=.85, label="log(1 + raw contact count)")
    fig.savefig(args.output, dpi=240)


if __name__ == "__main__":
    main()
