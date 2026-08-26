"""QC Module: Quantitative evaluation of CNV decoupling, phasing concordance, and posterior uncertainty."""

from __future__ import annotations

import os
from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats

from .ssm import SSMResults


def compute_qc_metrics(
    results: SSMResults,
    extra_data: dict,
) -> Dict[str, float]:
    """Compute comprehensive QC metrics evaluating CNV decoupling and compartment fidelity.

    Metrics include:
    - corr_x_cnv_pearson, corr_x_cnv_spearman: Correlation between latent score x and CNV (lower is better).
    - corr_x_phasing_pearson: Correlation between latent score x and gene density/GC (higher is better).
    - corr_pc1_cnv_pearson: Correlation between raw PC1 and CNV (benchmark).
    - corr_pc1_phasing_pearson: Correlation between raw PC1 and gene density/GC.
    - cnv_contamination_reduction: |corr_pc1_cnv| - |corr_x_cnv| (positive means CNV artifact was successfully reduced).
    - fraction_A, fraction_B, fraction_U: Proportions of discrete calls.
    - mean_posterior_sd, median_posterior_sd: Average uncertainty in state estimation.
    """
    valid_mask = extra_data["valid_mask"]
    x = results.state_score
    c = results.cnv_deviation
    phasing = extra_data.get("phasing_track")
    y_obs = extra_data.get("y_obs")

    qc = {
        "chrom": results.chrom,
        "n_total_bins": len(results.bins),
        "n_valid_bins": int(np.sum(valid_mask)),
        "valid_bin_fraction": float(np.mean(valid_mask)),
        "model_type": results.model_type,
        "iterations": results.iterations,
        "converged": int(results.converged),
        "log_likelihood": float(results.log_likelihood),
    }

    # 1. Correlation between latent score x and CNV deviation c
    mask_xc = valid_mask & np.isfinite(x) & np.isfinite(c)
    if np.sum(mask_xc) > 10 and np.std(c[mask_xc]) > 1e-6:
        r_xc_p, _ = stats.pearsonr(x[mask_xc], c[mask_xc])
        r_xc_s, _ = stats.spearmanr(x[mask_xc], c[mask_xc])
        qc["corr_x_cnv_pearson"] = float(r_xc_p)
        qc["corr_x_cnv_spearman"] = float(r_xc_s)
    else:
        qc["corr_x_cnv_pearson"] = 0.0
        qc["corr_x_cnv_spearman"] = 0.0

    # 2. Correlation between raw PC1 / PC2 and CNV
    if y_obs is not None and np.sum(mask_xc) > 10 and np.std(c[mask_xc]) > 1e-6:
        pc1 = y_obs[:, 0]
        mask_pc1_c = valid_mask & np.isfinite(pc1) & np.isfinite(c)
        if np.sum(mask_pc1_c) > 10:
            r_pc1_c, _ = stats.pearsonr(pc1[mask_pc1_c], c[mask_pc1_c])
            qc["corr_pc1_cnv_pearson"] = float(r_pc1_c)
            qc["cnv_contamination_reduction"] = float(abs(r_pc1_c) - abs(qc["corr_x_cnv_pearson"]))
        if y_obs.shape[1] > 1:
            pc2 = y_obs[:, 1]
            mask_pc2_c = valid_mask & np.isfinite(pc2) & np.isfinite(c)
            if np.sum(mask_pc2_c) > 10:
                r_pc2_c, _ = stats.pearsonr(pc2[mask_pc2_c], c[mask_pc2_c])
                qc["corr_pc2_cnv_pearson"] = float(r_pc2_c)

    # 3. Phasing concordance (Gene density / GC correlation)
    if phasing is not None:
        mask_xp = valid_mask & np.isfinite(x) & np.isfinite(phasing)
        if np.sum(mask_xp) > 10 and np.std(phasing[mask_xp]) > 1e-6:
            r_xp_p, _ = stats.pearsonr(x[mask_xp], phasing[mask_xp])
            r_xp_s, _ = stats.spearmanr(x[mask_xp], phasing[mask_xp])
            qc["corr_x_phasing_pearson"] = float(r_xp_p)
            qc["corr_x_phasing_spearman"] = float(r_xp_s)

        if y_obs is not None:
            pc1 = y_obs[:, 0]
            mask_pc1_p = valid_mask & np.isfinite(pc1) & np.isfinite(phasing)
            if np.sum(mask_pc1_p) > 10 and np.std(phasing[mask_pc1_p]) > 1e-6:
                r_pc1_p, _ = stats.pearsonr(pc1[mask_pc1_p], phasing[mask_pc1_p])
                qc["corr_pc1_phasing_pearson"] = float(r_pc1_p)

    # 4. Discrete State Calls Distribution
    valid_states = results.discrete_state[valid_mask]
    n_valid = len(valid_states)
    if n_valid > 0:
        qc["fraction_A"] = float(np.sum(valid_states == "A") / n_valid)
        qc["fraction_B"] = float(np.sum(valid_states == "B") / n_valid)
        qc["fraction_U"] = float(np.sum(valid_states == "U") / n_valid)
        valid_conf = results.confidence[valid_mask]
        qc["fraction_high_confidence"] = float(np.sum(valid_conf == "high") / n_valid)

    # 5. Posterior Uncertainty Summary
    valid_sd = results.posterior_sd[valid_mask & np.isfinite(results.posterior_sd)]
    if len(valid_sd) > 0:
        qc["mean_posterior_sd"] = float(np.mean(valid_sd))
        qc["median_posterior_sd"] = float(np.median(valid_sd))

    return qc


def export_qc_tsv(
    qc_dict: Dict[str, float],
    out_path: str,
) -> None:
    """Export QC dictionary to TSV format."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame([{"metric": k, "value": v} for k, v in qc_dict.items()])
    df.to_csv(out_path, sep="\t", index=False)
