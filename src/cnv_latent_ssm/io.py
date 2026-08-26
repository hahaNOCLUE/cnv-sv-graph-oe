"""IO Module: Exporters for compartment tracks, states, CNVs, and model parameters."""

from __future__ import annotations

import os
from typing import Optional
import numpy as np
import pandas as pd

from .ssm import SSMResults


def calder_style_rank_score(values: np.ndarray) -> np.ndarray:
    """Map finite chromosome scores to [-1, 1] by average empirical rank."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    n = int(np.sum(valid))
    if n == 0:
        return out
    if n == 1:
        out[valid] = 0.0
        return out
    ranks = pd.Series(values[valid]).rank(method="average").to_numpy()
    out[valid] = 2.0 * (ranks - 1.0) / (n - 1.0) - 1.0
    return out


def export_compartment_bedgraph(
    results: SSMResults,
    out_path: str,
) -> None:
    """Export continuous compartment score to 4-column bedGraph: chr, start, end, score."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame({
        "chrom": results.bins["chrom"],
        "start": results.bins["start"].astype(int),
        "end": results.bins["end"].astype(int),
        "score": np.round(results.state_score, 6),
    })
    # Filter out NaNs for standard bedGraph
    df_valid = df.dropna(subset=["score"]).copy()
    df_valid.to_csv(out_path, sep="\t", header=False, index=False)


def export_rank_compartment_bedgraph(results: SSMResults, out_path: str) -> None:
    """Export CALDER-style chromosome rank score in the interval [-1, 1]."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    rank_score = calder_style_rank_score(results.state_score)
    df = pd.DataFrame({
        "chrom": results.bins["chrom"],
        "start": results.bins["start"].astype(int),
        "end": results.bins["end"].astype(int),
        "score": np.round(rank_score, 6),
    }).dropna(subset=["score"])
    df.to_csv(out_path, sep="\t", header=False, index=False)


def export_local_artifact_bedgraph(results: SSMResults, out_path: str) -> None:
    """Export the fast local state from a two-scale model."""
    if results.local_artifact_score is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame({
        "chrom": results.bins["chrom"],
        "start": results.bins["start"].astype(int),
        "end": results.bins["end"].astype(int),
        "score": np.round(results.local_artifact_score, 6),
    }).dropna(subset=["score"])
    df.to_csv(out_path, sep="\t", header=False, index=False)


def export_compartment_state_bed(
    results: SSMResults,
    out_path: str,
) -> None:
    """Export discrete compartment calls with uncertainty and posterior details to BED format."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame({
        "chrom": results.bins["chrom"],
        "start": results.bins["start"].astype(int),
        "end": results.bins["end"].astype(int),
        "state_score": np.round(results.state_score, 6),
        "posterior_sd": np.round(results.posterior_sd, 6),
        "state": results.discrete_state,
        "confidence": results.confidence,
        "p_active": np.round(results.p_active, 6),
    })
    df.to_csv(out_path, sep="\t", index=False)


def export_cnv_bedgraph(
    results: SSMResults,
    out_path: str,
) -> None:
    """Export CNV deviation c_i to bedGraph format."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame({
        "chrom": results.bins["chrom"],
        "start": results.bins["start"].astype(int),
        "end": results.bins["end"].astype(int),
        "cnv_deviation": np.round(results.cnv_deviation, 6),
    })
    df_valid = df.dropna(subset=["cnv_deviation"]).copy()
    df_valid.to_csv(out_path, sep="\t", header=False, index=False)


def export_parameters_tsv(
    results: SSMResults,
    out_path: str,
) -> None:
    """Export estimated SSM parameters (F, Q, R, H, G, logLik) to a TSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    records = []

    # Model metadata
    records.append({"category": "meta", "param": "model_type", "index": "0", "value": results.model_type})
    records.append({"category": "meta", "param": "log_likelihood", "index": "0", "value": f"{results.log_likelihood:.6f}"})
    records.append({"category": "meta", "param": "iterations", "index": "0", "value": str(results.iterations)})
    records.append({"category": "meta", "param": "converged", "index": "0", "value": str(results.converged)})

    # Transition F
    if results.F is not None:
        for i in range(results.F.shape[0]):
            for j in range(results.F.shape[1]):
                records.append({
                    "category": "transition_F",
                    "param": f"F_{i+1}_{j+1}",
                    "index": f"({i+1},{j+1})",
                    "value": f"{results.F[i, j]:.6f}",
                })

    # Process noise Q
    if results.Q is not None:
        for i in range(results.Q.shape[0]):
            records.append({
                "category": "process_noise_Q",
                "param": f"Q_{i+1}_{i+1}",
                "index": f"{i+1}",
                "value": f"{results.Q[i, i]:.6f}",
            })

    # Observation loadings H
    if results.H is not None:
        for k in range(results.H.shape[0]):
            for s in range(results.H.shape[1]):
                records.append({
                    "category": "compartment_loading_H",
                    "param": f"H_PC{k+1}_s{s+1}",
                    "index": f"PC{k+1}",
                    "value": f"{results.H[k, s]:.6f}",
                })

    # CNV effect G
    if results.G is not None:
        for k in range(results.G.shape[0]):
            records.append({
                "category": "cnv_effect_G",
                "param": f"G_PC{k+1}",
                "index": f"PC{k+1}",
                "value": f"{results.G[k, 0]:.6f}",
            })

    # Observation noise R
    if results.R is not None:
        for k in range(results.R.shape[0]):
            records.append({
                "category": "observation_noise_R",
                "param": f"R_PC{k+1}",
                "index": f"PC{k+1}",
                "value": f"{results.R[k, k]:.6f}",
            })

    df = pd.DataFrame(records)
    df.to_csv(out_path, sep="\t", index=False)
