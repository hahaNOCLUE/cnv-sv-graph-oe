"""Feature Construction Module: Hi-C/Micro-C contact matrix processing, O/E normalization, correlation, and PCA/SVD compression."""

from __future__ import annotations

import logging
import warnings
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def extract_chrom_matrix_and_bins(
    cooler_path: str,
    chrom: str,
    resolution: Optional[int] = None,
    balance: bool = False,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract intra-chromosomal contact matrix and bin annotations from a cooler/mcool file.

    Parameters
    ----------
    cooler_path : str
        Path to .cool or .mcool file (or .mcool::/resolutions/50000).
    chrom : str
        Chromosome name (e.g. 'chr8').
    resolution : int, optional
        Target resolution in bp (if .mcool is provided without resolution suffix).
    balance : bool
        Whether to use balanced weights or raw unbalanced counts. Default False (raw).
    """
    import cooler

    uri = cooler_path
    if "::" not in uri and resolution is not None:
        uri = f"{cooler_path}::/resolutions/{resolution}"

    clr = cooler.Cooler(uri)
    
    # Standardize chromosome name
    if chrom not in clr.chromnames:
        alt_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        if alt_chrom in clr.chromnames:
            chrom = alt_chrom
        else:
            raise KeyError(f"Chromosome '{chrom}' not found in cooler: {clr.chromnames}")

    # Fetch bins
    bins_df = clr.bins().fetch(chrom).copy().reset_index(drop=True)

    # Fetch matrix
    mat_selector = clr.matrix(balance=balance)
    mat = mat_selector.fetch(chrom)

    # Replace NaNs / Infs
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat, bins_df


def identify_valid_bins(
    raw_matrix: np.ndarray,
    bins_df: Optional[pd.DataFrame] = None,
    clr_weight_name: str = "weight",
    require_balancing_weight: bool = True,
) -> np.ndarray:
    """Identify valid (unmasked) bins matching cooltools filtering criteria.

    A bin is valid if:
    1. It has non-zero marginal sum in the contact matrix (sum(axis=0) > 0).
    2. By default, the existing ICE mask is applied even when the matrix values
       are raw counts: the requested cooler weight must be finite and positive.
    """
    row_sums = np.sum(raw_matrix, axis=1)
    valid = (row_sums > 0) & np.isfinite(row_sums)

    if require_balancing_weight and bins_df is not None and clr_weight_name in bins_df.columns:
        weights = pd.to_numeric(bins_df[clr_weight_name], errors="coerce").to_numpy()
        weight_valid = np.isfinite(weights) & (weights > 0)
        valid = valid & weight_valid

    return valid


def compute_observed_over_expected(
    matrix: np.ndarray,
    valid_mask: np.ndarray,
    max_distance_bins: Optional[int] = None,
) -> np.ndarray:
    """Compute distance-dependent Observed/Expected (O/E) matrix."""
    n = matrix.shape[0]
    if max_distance_bins is None:
        max_distance_bins = n

    oe_mat = np.zeros_like(matrix, dtype=float)
    valid_grid = np.outer(valid_mask, valid_mask)

    # Calculate expected contact frequency at each diagonal distance d
    expected = np.zeros(n, dtype=float)
    for d in range(n):
        if d >= max_distance_bins:
            break
        diag_vals = np.diag(matrix, k=d)
        diag_valid = np.diag(valid_grid, k=d)
        if np.sum(diag_valid) > 0:
            expected[d] = np.mean(diag_vals[diag_valid])
        else:
            expected[d] = 0.0

    # Avoid divide by zero
    for d in range(n):
        if d >= max_distance_bins or expected[d] <= 1e-12:
            continue
        exp_val = expected[d]
        i_indices = np.arange(n - d)
        j_indices = i_indices + d

        pair_valid = valid_mask[i_indices] & valid_mask[j_indices]
        valid_i = i_indices[pair_valid]
        valid_j = j_indices[pair_valid]

        oe_vals = matrix[valid_i, valid_j] / exp_val
        oe_mat[valid_i, valid_j] = oe_vals
        oe_mat[valid_j, valid_i] = oe_vals

    return oe_mat


def compute_sv_distance_mixture_oe(
    matrix: np.ndarray,
    valid_mask: np.ndarray,
    sv_edges: list,
    max_sv_hops: Optional[int] = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    """O/E with confidence-weighted multi-SV effective distances.

    ``sv_edges`` contains ``(u, v, pi)`` tuples, where ``pi`` combines SV-call
    confidence and endpoint dosage.  Paths can optionally use up to
    ``max_sv_hops`` SV
    edges, which lets nearby/compound SVs jointly affect a contact block.
    Linear genomic segments remain unit-cost edges.  Among equally short paths,
    the path with the largest *bottleneck* SV weight is retained.  The
    bottleneck, rather than a product of edge weights, avoids treating nearby
    calls from one complex event as independent subclones.

    The hop cap prevents a chain of unrelated calls from creating implausible
    chromosome-wide shortcuts.  Set ``return_diagnostics`` to obtain reference
    and effective distances, SV-hop counts, and the retained mixture weight.
    """
    warnings.warn(
        "compute_sv_distance_mixture_oe is a legacy shortest-path heuristic; "
        "use graph_expected.compute_copy_flow_additive_oe for oriented JCN flows",
        DeprecationWarning, stacklevel=2,
    )
    if max_sv_hops is not None and max_sv_hops < 1:
        raise ValueError("max_sv_hops must be at least 1")

    n = matrix.shape[0]
    valid_grid = np.outer(valid_mask, valid_mask)
    expected = np.zeros(n, dtype=float)
    for d in range(n):
        values = np.diag(matrix, k=d)
        keep = np.diag(valid_grid, k=d)
        if np.any(keep):
            expected[d] = np.mean(values[keep])

    idx = np.arange(n)
    d_ref = np.abs(idx[:, None] - idx[None, :]).astype(np.int32)
    d_sv = d_ref.copy()
    hops = np.zeros((n, n), dtype=np.int16)
    strength = np.ones((n, n), dtype=np.float32)

    cleaned_edges = []
    for u, v, pi in sv_edges:
        u, v = int(u), int(v)
        if not (0 <= u < n and 0 <= v < n) or u == v:
            continue
        cleaned_edges.append((u, v, float(np.clip(pi, 0.0, 1.0))))

    # With no explicit cap, n-1 relaxation passes cover every simple path.
    # A finite cap remains useful as a sensitivity diagnostic, but is no longer
    # silently fixed at three hops.
    hop_limit = (max(n - 1, 1) if max_sv_hops is None else max_sv_hops)
    for _ in range(hop_limit):
        changed = False
        for u, v, pi in cleaned_edges:
            for left, right in ((u, v), (v, u)):
                candidate_distance = (
                    d_sv[:, left, None] + 1 + d_sv[right, None, :]
                )
                candidate_hops = (
                    hops[:, left, None] + 1 + hops[right, None, :]
                )
                candidate_strength = np.minimum(
                    np.minimum(strength[:, left, None], pi),
                    strength[right, None, :],
                )
                usable = candidate_hops <= hop_limit
                better = usable & (
                    (candidate_distance < d_sv)
                    | (
                        (candidate_distance == d_sv)
                        & (candidate_hops > 0)
                        & (candidate_strength > strength)
                    )
                )
                if np.any(better):
                    d_sv[better] = candidate_distance[better]
                    hops[better] = candidate_hops[better]
                    strength[better] = candidate_strength[better]
                    changed = True
        if not changed:
            break

    shortened = (d_sv < d_ref) & (hops > 0)
    mixture = np.where(shortened, strength, 0.0).astype(np.float32)

    expected_ref = expected[d_ref]
    expected_sv = expected[d_sv]
    mixed_expected = (1.0 - mixture) * expected_ref + mixture * expected_sv
    usable = valid_grid & (mixed_expected > 1e-12)
    oe = np.zeros_like(matrix, dtype=float)
    oe[usable] = matrix[usable] / mixed_expected[usable]
    if not return_diagnostics:
        return oe
    return oe, {
        "reference_distance": d_ref,
        "effective_distance": d_sv,
        "sv_hops": hops,
        "mixture_weight": mixture,
    }


def compute_interaction_profile_correlation(
    oe_matrix: np.ndarray,
    valid_mask: np.ndarray,
    log_transform: bool = True,
    winsorize_quantile: float = 0.999,
) -> np.ndarray:
    """Compute whole-chromosome interaction-profile correlation matrix S."""
    n = oe_matrix.shape[0]
    mat = oe_matrix.copy()

    if log_transform:
        mat = np.log1p(mat)

    # Winsorize extreme values on valid submatrix
    valid_vals = mat[valid_mask][:, valid_mask]
    if len(valid_vals) > 0:
        high = np.quantile(valid_vals, winsorize_quantile)
        mat = np.clip(mat, 0.0, high)

    # Compute correlation on valid columns
    S = np.zeros((n, n), dtype=float)
    valid_sub = mat[valid_mask][:, valid_mask]

    # Center rows
    means = np.mean(valid_sub, axis=1, keepdims=True)
    centered = valid_sub - means
    stds = np.std(centered, axis=1, keepdims=True)
    stds[stds < 1e-12] = 1.0
    normalized = centered / stds

    # Pearson correlation via matrix multiplication
    # np.std above uses ddof=0, hence each standardized row has squared norm n.
    corr_valid = np.dot(normalized, normalized.T) / normalized.shape[1]
    corr_valid = np.clip(corr_valid, -1.0, 1.0)
    np.fill_diagonal(corr_valid, 1.0)

    # Fill back into full size matrix
    valid_indices = np.where(valid_mask)[0]
    S[np.ix_(valid_indices, valid_indices)] = corr_valid

    return S


def extract_pca_observation_features(
    correlation_matrix: np.ndarray,
    valid_mask: np.ndarray,
    n_components: int = 10,
) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Extract top K principal components from correlation matrix S as observation vectors y_i.

    Parameters
    ----------
    correlation_matrix : np.ndarray
        Correlation matrix S (size N x N).
    valid_mask : np.ndarray
        Boolean mask of valid bins (size N).
    n_components : int
        Number of principal components K (default 10).

    Returns
    -------
    y : np.ndarray
        Observation matrix of shape (N, K). Invalid bins contain NaNs.
    explained_variance_ratio : np.ndarray
        Explained variance ratio for each component.
    pca_obj : PCA
        Fitted scikit-learn PCA object.
    """
    n = len(valid_mask)
    y = np.full((n, n_components), np.nan, dtype=float)

    valid_sub = correlation_matrix[valid_mask][:, valid_mask]
    n_valid = np.sum(valid_mask)
    k = min(n_components, n_valid - 1)

    pca = PCA(n_components=k)
    pcs_valid = pca.fit_transform(valid_sub)

    # If k < n_components, pad with zeros
    if k < n_components:
        padded = np.zeros((n_valid, n_components))
        padded[:, :k] = pcs_valid
        pcs_valid = padded

    y[valid_mask] = pcs_valid
    return y, pca.explained_variance_ratio_, pca
