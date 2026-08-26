"""Plotting Module: Publication-quality multi-track visualization of CNV-aware compartment modeling."""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ssm import SSMResults


def plot_ssm_summary(
    results: SSMResults,
    extra_data: dict,
    out_prefix: str,
    title: Optional[str] = None,
) -> str:
    """Generate multi-panel summary plot comparing raw PCs, CNV, SSM score, and loadings."""
    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)
    out_png = f"{out_prefix}.summary.png"

    chrom = results.chrom
    starts_mb = results.bins["start"].to_numpy() / 1e6
    ends_mb = results.bins["end"].to_numpy() / 1e6
    pos_mb = (starts_mb + ends_mb) / 2.0
    n_bins = len(pos_mb)

    valid_mask = extra_data["valid_mask"]
    y_obs = extra_data.get("y_obs")
    phasing = extra_data.get("phasing_track")
    c_i = results.cnv_deviation
    x_score = results.state_score
    sd = results.posterior_sd

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=False, gridspec_kw={"height_ratios": [1.2, 1.0, 1.4, 0.4, 0.8, 1.2]})

    # ----------------- Track 1: Observation PCs (PC1, PC2, PC3) -----------------
    ax0 = axes[0]
    if y_obs is not None:
        if y_obs.shape[1] >= 1:
            ax0.plot(pos_mb[valid_mask], y_obs[valid_mask, 0], label="PC1", color="#1f77b4", lw=1.2, alpha=0.85)
        if y_obs.shape[1] >= 2:
            ax0.plot(pos_mb[valid_mask], y_obs[valid_mask, 1], label="PC2", color="#2ca02c", lw=1.0, alpha=0.7)
        if y_obs.shape[1] >= 3:
            ax0.plot(pos_mb[valid_mask], y_obs[valid_mask, 2], label="PC3", color="#ff7f0e", lw=1.0, alpha=0.7)
    ax0.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax0.set_ylabel("PCA Features\n(Input y_i)", fontsize=10)
    ax0.set_title(title or f"CNV-aware Latent Chromatin State Model — {chrom}", fontsize=13, fontweight="bold")
    ax0.legend(loc="upper right", frameon=True, fontsize=8, ncol=3)
    ax0.set_xlim(pos_mb[0], pos_mb[-1])
    ax0.grid(True, linestyle=":", alpha=0.5)

    # ----------------- Track 2: CNV Deviation c_i -----------------
    ax1 = axes[1]
    ax1.plot(pos_mb, c_i, color="#d62728", lw=1.5, label="CNV deviation c_i = log2(CN/P)")
    ax1.fill_between(pos_mb, 0, c_i, where=(c_i > 0), color="#d62728", alpha=0.25)
    ax1.fill_between(pos_mb, 0, c_i, where=(c_i < 0), color="#1f77b4", alpha=0.25)
    ax1.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax1.set_ylabel("CNV Deviation\nlog2(CN / P)", fontsize=10)
    ax1.legend(loc="upper right", frameon=True, fontsize=8)
    ax1.set_xlim(pos_mb[0], pos_mb[-1])
    ax1.grid(True, linestyle=":", alpha=0.5)

    # ----------------- Track 3: Continuous Latent SSM Score x_i with Uncertainty -----------------
    ax2 = axes[2]
    # Keep NaNs in array so matplotlib creates visual gaps across deleted/unmapped regions
    x_plot = np.where(valid_mask & np.isfinite(x_score), x_score, np.nan)
    sd_plot = np.where(valid_mask & np.isfinite(sd), sd, np.nan)

    ax2.plot(pos_mb, x_plot, color="#800080", lw=1.5, label="Latent Score E[x_i | Y, CN]")
    valid_ci = np.isfinite(x_plot) & np.isfinite(sd_plot)
    ax2.fill_between(
        pos_mb,
        x_plot - 1.96 * sd_plot,
        x_plot + 1.96 * sd_plot,
        where=valid_ci,
        color="#800080",
        alpha=0.2,
        label="95% Posterior CI",
    )
    ax2.axhline(0, color="black", linestyle="--", lw=1.0)
    ax2.set_ylabel("Continuous SSM\nCompartment Score", fontsize=10)
    ax2.legend(loc="upper right", frameon=True, fontsize=8)
    ax2.set_xlim(pos_mb[0], pos_mb[-1])
    ax2.grid(True, linestyle=":", alpha=0.5)

    # ----------------- Track 4: Discrete States (A / B / U / Del) -----------------
    ax3 = axes[3]
    state_arr = np.zeros(n_bins)
    # A -> 1 (red), B -> -1 (blue), U -> 0 (light gray), Del/NA -> NaN (white/transparent)
    for i in range(n_bins):
        st = str(results.discrete_state[i])
        if st == "A":
            state_arr[i] = 1.0
        elif st == "B":
            state_arr[i] = -1.0
        elif st == "U":
            state_arr[i] = 0.0
        else:
            state_arr[i] = np.nan

    cmap = matplotlib.colors.ListedColormap(["#2166ac", "#d9d9d9", "#b2182b"])
    cmap.set_bad(color="#ffffff")
    norm = matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    masked_states = np.ma.masked_invalid(state_arr[np.newaxis, :])
    ax3.imshow(
        masked_states,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[pos_mb[0], pos_mb[-1], 0, 1],
    )
    ax3.set_yticks([])
    ax3.set_ylabel("State\n(A / U / B)", fontsize=10)
    ax3.set_xlim(pos_mb[0], pos_mb[-1])

    # ----------------- Track 5: Phasing Track (Gene density / GC) -----------------
    ax4 = axes[4]
    if phasing is not None:
        p_valid = phasing[valid_mask]
        ax4.plot(pos_mb[valid_mask], p_valid, color="#2b8cbe", lw=1.0, label="GC / Gene Density Activity")
        ax4.axhline(0, color="gray", linestyle="--", lw=0.8)
    ax4.set_ylabel("Phasing\nActivity", fontsize=10)
    ax4.set_xlim(pos_mb[0], pos_mb[-1])
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.set_xlabel("Genomic Position (Mb)", fontsize=11)

    # ----------------- Subplot 6: Loadings H vs CNV effects G across PCs -----------------
    ax5 = axes[5]
    if results.H is not None and results.G is not None:
        K = results.H.shape[0]
        pc_idx = np.arange(1, K + 1)
        width = 0.35

        h_vals = results.H[:, 0]
        g_vals = results.G[:, 0]

        rects1 = ax5.bar(pc_idx - width / 2, h_vals, width, label="Compartment Loading H (Signal)", color="#984ea3")
        rects2 = ax5.bar(pc_idx + width / 2, g_vals, width, label="CNV Effect G (Artifact)", color="#e41a1c")

        ax5.set_xticks(pc_idx)
        ax5.set_xticklabels([f"PC{k}" for k in pc_idx], fontsize=9)
        ax5.set_ylabel("Estimated Loading / Effect", fontsize=10)
        ax5.set_xlabel("Principal Components (y_i)", fontsize=10)
        ax5.set_title("SSM Parameter Decomposition: Compartment Loading H vs CNV Effect G", fontsize=11, fontweight="bold")
        ax5.axhline(0, color="black", linestyle="-", lw=0.8)
        ax5.legend(loc="upper right", frameon=True, fontsize=8)
        ax5.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png
