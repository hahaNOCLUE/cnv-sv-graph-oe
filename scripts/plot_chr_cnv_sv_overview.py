#!/usr/bin/env python3
"""Plot chromosome copy number and balanced intra/interchromosomal SVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc
import numpy as np
import pandas as pd


ROOT = Path("/home/dell/a1/microc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrom", default="chrX")
    parser.add_argument(
        "--cnv",
        type=Path,
        default=ROOT / "result_2/compartment/CNVkit_F1_50kb/F1_50kb.cbs.cns",
    )
    parser.add_argument(
        "--junctions",
        type=Path,
        default=ROOT
        / "code/cnv_latent_ssm/examples/F1_chrX/input_tables/F1_chrX.balanced_junction_cn.tsv",
    )
    parser.add_argument("--ploidy", type=float, default=2.0)
    parser.add_argument("--centromere-start", type=float, default=58_055_932)
    parser.add_argument("--centromere-end", type=float, default=63_829_925)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "result_2/compartment/chrX_graph_aware_caic_response/chrX.cnv_sv_overview.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cnv = pd.read_csv(args.cnv, sep="\t")
    cnv = cnv.loc[cnv["chromosome"].astype(str).eq(args.chrom)].copy()
    if cnv.empty:
        raise ValueError(f"No CNV segments found for {args.chrom}")
    cnv["copy_number"] = args.ploidy * np.exp2(pd.to_numeric(cnv["log2"]))

    sv = pd.read_csv(args.junctions, sep="\t")
    sv = sv.loc[sv["chrom1"].eq(args.chrom) | sv["chrom2"].eq(args.chrom)].copy()
    sv["junction_cn"] = pd.to_numeric(sv["junction_cn"], errors="coerce").fillna(0)
    sv["eaglec2_probability"] = pd.to_numeric(
        sv["eaglec2_probability"], errors="coerce"
    ).fillna(0)

    chrom_end = float(cnv["end"].max())
    fig, (ax_cn, ax_sv) = plt.subplots(
        2,
        1,
        figsize=(16, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 2.1], "hspace": 0.08},
    )

    for row in cnv.itertuples():
        ax_cn.plot(
            [row.start / 1e6, row.end / 1e6],
            [row.copy_number, row.copy_number],
            color="#2166ac",
            lw=4,
            solid_capstyle="butt",
        )
        ax_cn.plot(
            [row.end / 1e6, row.end / 1e6],
            [0, row.copy_number],
            color="#2166ac",
            lw=0.35,
            alpha=0.35,
        )
    ax_cn.axhline(args.ploidy, color="0.35", lw=1, ls="--", label=f"CN={args.ploidy:g}")
    ax_cn.set_ylabel("copy number")
    ax_cn.set_ylim(0, max(3.0, float(cnv["copy_number"].max()) * 1.18))
    ax_cn.legend(frameon=False, loc="upper right")
    ax_cn.set_title(f"F1 {args.chrom}: CNV and balanced EagleC2 SV overview")

    intra = sv.loc[sv["chrom1"].eq(args.chrom) & sv["chrom2"].eq(args.chrom)].copy()
    external = sv.loc[~(sv["chrom1"].eq(args.chrom) & sv["chrom2"].eq(args.chrom))].copy()
    max_span = max((abs(r.pos2 - r.pos1) for r in intra.itertuples()), default=1.0)
    max_jcn = max(float(sv["junction_cn"].max()), 1e-6)

    cent_start = args.centromere_start / 1e6
    cent_end = args.centromere_end / 1e6
    for axis in (ax_cn, ax_sv):
        axis.axvspan(cent_start, cent_end, color="0.55", alpha=0.20, zorder=0)
    ax_cn.text(
        (cent_start + cent_end) / 2,
        0.97,
        f"centromere\n{cent_start:.1f}-{cent_end:.1f} Mb",
        transform=ax_cn.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color="0.30",
    )

    intra_breakpoints = sorted(
        set(intra["pos1"].astype(float) / 1e6) | set(intra["pos2"].astype(float) / 1e6)
    )
    external_breakpoints = []
    for row in external.itertuples():
        external_breakpoints.append(
            (row.pos1 if row.chrom1 == args.chrom else row.pos2) / 1e6
        )
    for axis in (ax_cn, ax_sv):
        for x in intra_breakpoints:
            axis.axvline(x, color="#b2182b", lw=0.65, ls="--", alpha=0.34, zorder=1)
        for x in external_breakpoints:
            axis.axvline(x, color="#542788", lw=0.85, ls=":", alpha=0.55, zorder=1)

    for row in intra.itertuples():
        x1, x2 = sorted((row.pos1 / 1e6, row.pos2 / 1e6))
        width = x2 - x1
        height = 8 + 42 * np.sqrt(abs(row.pos2 - row.pos1) / max_span)
        color = "#b2182b" if row.junction_cn >= 0.1 else "#ef8a62"
        lw = 0.8 + 4.0 * np.sqrt(row.junction_cn / max_jcn)
        alpha = 0.35 + 0.65 * row.eaglec2_probability
        ax_sv.add_patch(
            Arc(
                ((x1 + x2) / 2, 0),
                width=width,
                height=height,
                theta1=0,
                theta2=180,
                color=color,
                lw=lw,
                alpha=alpha,
            )
        )

    label_levels: dict[int, int] = {}
    for row in external.sort_values("pos1").itertuples():
        if row.chrom1 == args.chrom:
            x, partner = row.pos1 / 1e6, f"{row.chrom2}:{row.pos2 / 1e6:.1f}"
        else:
            x, partner = row.pos2 / 1e6, f"{row.chrom1}:{row.pos1 / 1e6:.1f}"
        bucket = int(x // 8)
        level = label_levels.get(bucket, 0)
        label_levels[bucket] = level + 1
        y = -7.0 - 6.0 * level
        direction = 1 if level % 2 == 0 else -1
        label_x = min(max(x + direction * (4.0 + 2.0 * level), 2.0), chrom_end / 1e6 - 2.0)
        size = 35 + 150 * np.sqrt(row.junction_cn / max_jcn)
        ax_sv.scatter(x, 0, marker="v", s=size, color="#542788", zorder=5)
        ax_sv.annotate(
            partner,
            xy=(x, -0.8),
            xytext=(label_x, y),
            arrowprops={"arrowstyle": "-", "color": "#542788", "lw": 0.8},
            color="#542788",
            ha="center",
            va="top",
            fontsize=8,
        )

    ax_sv.axhline(0, color="0.2", lw=1)
    ax_sv.set_xlim(0, chrom_end / 1e6)
    ax_sv.set_ylim(-34, 58)
    ax_sv.set_yticks([])
    ax_sv.set_xlabel(f"{args.chrom} position (Mb)")
    ax_sv.set_ylabel("SV arcs")
    ax_sv.legend(
        handles=[
            Line2D([0], [0], color="#b2182b", lw=3, label="intrachromosomal SV (width ~ JCN)"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor="#542788", markersize=9,
                   label="interchromosomal SV breakend"),
            Line2D([0], [0], color="#ef8a62", lw=1, label="low-flow JCN < 0.1"),
            Line2D([0], [0], color="0.55", lw=7, alpha=0.35, label="centromere"),
        ],
        frameon=False,
        loc="upper right",
    )
    ax_sv.text(
        0.01,
        0.97,
        f"{len(intra)} intra-chr SVs; {len(external)} inter-chr SVs",
        transform=ax_sv.transAxes,
        va="top",
        fontsize=10,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
