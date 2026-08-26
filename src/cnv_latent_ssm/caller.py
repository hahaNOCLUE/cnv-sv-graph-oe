"""Caller Module: High-level pipeline orchestrating contact loading, feature extraction, CNV deviation calculation, and SSM fitting."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .cnv import get_cnv_track
from .features import (
    compute_interaction_profile_correlation,
    compute_observed_over_expected,
    compute_sv_distance_mixture_oe,
    extract_chrom_matrix_and_bins,
    extract_pca_observation_features,
    identify_valid_bins,
)
from .ssm import CNVAwareSSM, SSMResults

logger = logging.getLogger(__name__)


def identify_transition_breakpoints(
    bins_df: pd.DataFrame,
    valid_mask: np.ndarray,
    cnv_deviation: np.ndarray,
    min_gap_bins: int = 3,
    decouple_cnv_jumps: bool = False,
    cnv_jump_threshold: float = 0.4,
) -> np.ndarray:
    """Find boundaries at which the latent Markov chain should be restarted.

    Short runs of low-coverage bins are deliberately bridged.  They are common in
    heterochromatin and are missing observations, not evidence for a compartment
    boundary.  Likewise, a CNV segment boundary is handled by the observation
    term ``G*c`` and is not a state boundary unless explicitly requested.
    """
    n_bins = len(bins_df)
    breakpoints = np.zeros(n_bins, dtype=bool)
    if n_bins == 0:
        return breakpoints
    breakpoints[0] = True

    gap_run = 0
    for t in range(1, n_bins):
        if not valid_mask[t - 1]:
            gap_run += 1

        if valid_mask[t]:
            coordinate_gap = bins_df.iloc[t]["start"] != bins_df.iloc[t - 1]["end"]
            after_long_gap = gap_run >= max(min_gap_bins, 1)
            cnv_jump = (
                decouple_cnv_jumps
                and np.isfinite(cnv_deviation[t])
                and np.isfinite(cnv_deviation[t - 1])
                and abs(cnv_deviation[t] - cnv_deviation[t - 1]) >= cnv_jump_threshold
            )
            if coordinate_gap or after_long_gap or cnv_jump:
                breakpoints[t] = True
            gap_run = 0

    return breakpoints


def load_sv_endpoint_breakpoints(
    sv_file: str,
    chrom: str,
    bins_df: pd.DataFrame,
) -> Tuple[np.ndarray, list]:
    """Map intra-chromosomal EagleC/EagleC2 SV endpoints to bin boundaries."""
    sv = pd.read_csv(sv_file, sep=r"\s+", comment="#")
    required = {"chrom1", "pos1", "chrom2", "pos2"}
    if not required.issubset(sv.columns):
        raise ValueError(f"SV file must contain columns {sorted(required)}")
    target = chrom if chrom.startswith("chr") else f"chr{chrom}"
    norm1 = sv["chrom1"].astype(str).map(lambda x: x if x.startswith("chr") else f"chr{x}")
    norm2 = sv["chrom2"].astype(str).map(lambda x: x if x.startswith("chr") else f"chr{x}")
    sv = sv[(norm1 == target) & (norm2 == target)].copy()

    starts = bins_df["start"].to_numpy(dtype=int)
    breakpoints = np.zeros(len(bins_df), dtype=bool)
    edges = []
    for row in sv.itertuples(index=False):
        i = int(np.clip(np.searchsorted(starts, int(row.pos1), side="left"), 0, len(starts) - 1))
        j = int(np.clip(np.searchsorted(starts, int(row.pos2), side="left"), 0, len(starts) - 1))
        breakpoints[i] = True
        breakpoints[j] = True
        edges.append((i, j))
    return breakpoints, edges


def load_sv_distance_edges(
    sv_file: str, chrom: str, bins_df: pd.DataFrame, cn_relative: np.ndarray,
) -> list:
    """Create single-SV distance edges with confidence/CN mixture fractions."""
    sv = pd.read_csv(sv_file, sep=r"\s+", comment="#")
    target = chrom if chrom.startswith("chr") else f"chr{chrom}"
    c1 = sv["chrom1"].astype(str).map(lambda x: x if x.startswith("chr") else f"chr{x}")
    c2 = sv["chrom2"].astype(str).map(lambda x: x if x.startswith("chr") else f"chr{x}")
    sv = sv[(c1 == target) & (c2 == target)].copy()
    starts = bins_df["start"].to_numpy(dtype=int)
    probability_cols = [c for c in ["++", "+-", "-+", "--"] if c in sv.columns]
    edges = []
    for _, row in sv.iterrows():
        i = int(np.clip(np.searchsorted(starts, int(row["pos1"]), side="left"), 0, len(starts)-1))
        j = int(np.clip(np.searchsorted(starts, int(row["pos2"]), side="left"), 0, len(starts)-1))
        confidence = max(float(row[c]) for c in probability_cols) if probability_cols else 1.0
        dosage = float(np.sqrt(max(cn_relative[i], 0.0) * max(cn_relative[j], 0.0)))
        pi = confidence * dosage / (1.0 + dosage)
        edges.append((i, j, float(np.clip(pi, 0.0, 0.8))))
    return edges


def weight_sv_edges_by_cnv(
    edges: list,
    cn_relative: np.ndarray,
    min_weight: float = 0.25,
    max_weight: float = 4.0,
) -> Tuple[list, pd.DataFrame]:
    """Estimate relative SV-edge dosage from endpoint segment copy numbers."""
    dosages = [
        float(np.sqrt(max(cn_relative[i], 0.0) * max(cn_relative[j], 0.0)))
        for i, j in edges
    ]
    positive = np.asarray([d for d in dosages if np.isfinite(d) and d > 0])
    scale = float(np.median(positive)) if len(positive) else 1.0
    weighted, rows = [], []
    for (i, j), dosage in zip(edges, dosages):
        relative_weight = float(np.clip(dosage / scale, min_weight, max_weight))
        weighted.append((i, j, relative_weight))
        rows.append({
            "bin1": i, "bin2": j,
            "endpoint1_relative_cn": float(cn_relative[i]),
            "endpoint2_relative_cn": float(cn_relative[j]),
            "sv_cn_dosage_proxy": dosage,
            "relative_edge_weight": relative_weight,
        })
    return weighted, pd.DataFrame(rows)


def build_contact_knn_graph(
    correlation_matrix: np.ndarray,
    valid_mask: np.ndarray,
    k: int = 5,
    min_distance_bins: int = 40,
    min_correlation: float = 0.3,
    mutual: bool = True,
    signed: bool = False,
) -> list:
    """Build a distal contact-profile similarity graph without external tracks.

    Nodes are genomic bins and candidate edge weights are entries of the
    whole-chromosome O/E profile-correlation matrix.  By default an edge is
    retained only when the two bins select each other among their top-k distal
    neighbours.  This mutual-kNN criterion prevents a single unusual profile
    (for example, a CN/SV-driven hub) from being connected to many unrelated
    bins merely because it selected them in one direction.

    The retained weights are symmetrically degree-normalized and then scaled
    to median one.  Consequently ``contact_graph_strength`` has a comparable
    interpretation across chromosomes and choices of ``k``.
    """
    valid = np.flatnonzero(valid_mask)
    directed = {}
    for i in valid:
        candidates = valid[np.abs(valid - i) >= min_distance_bins]
        if not len(candidates):
            continue
        similarities = correlation_matrix[i, candidates]
        edge_scores = np.abs(similarities) if signed else similarities
        usable = np.isfinite(similarities) & (edge_scores >= min_correlation)
        candidates, similarities = candidates[usable], similarities[usable]
        edge_scores = edge_scores[usable]
        if not len(candidates):
            continue
        take = min(k, len(candidates))
        selected = np.argpartition(edge_scores, -take)[-take:]
        directed[int(i)] = {
            int(j): float(weight)
            for j, weight in zip(candidates[selected], similarities[selected])
        }

    edge_weights = {}
    for i, neighbours in directed.items():
        for j, weight_ij in neighbours.items():
            reverse = directed.get(j, {}).get(i)
            if mutual and reverse is None:
                continue
            if reverse is not None:
                relation = np.sign(weight_ij + reverse)
                weight = min(abs(weight_ij), abs(reverse))
            else:
                relation = np.sign(weight_ij)
                weight = abs(weight_ij)
            if not signed:
                relation = 1.0
            edge = (min(i, j), max(i, j))
            previous = edge_weights.get(edge)
            if previous is None or weight > previous[0]:
                edge_weights[edge] = (float(weight), float(relation))
    if not edge_weights:
        return []

    degree = {int(i): 0.0 for i in valid}
    for (i, j), (weight, _) in edge_weights.items():
        degree[i] += weight
        degree[j] += weight
    normalized = {
        edge: weight / np.sqrt(max(degree[edge[0]] * degree[edge[1]], 1e-12))
        for edge, (weight, _) in edge_weights.items()
    }
    scale = float(np.median(list(normalized.values())))
    return [
        (i, j, float(np.clip(weight / scale, 0.25, 4.0)), edge_weights[(i, j)][1])
        for (i, j), weight in normalized.items()
    ]


def load_phasing_track(
    phasing_file: str,
    chrom: str,
    bins_df: pd.DataFrame,
    col_name: Optional[str] = None,
) -> np.ndarray:
    """Load GC fraction, gene density, or activity track for chromosome bin alignment."""
    df = pd.read_csv(phasing_file, sep=r"\s+", comment="#")
    
    # Identify chrom column
    chrom_col = [c for c in df.columns if "chrom" in c.lower() or c == "chr"]
    if not chrom_col:
        chrom_col = df.columns[0]
    else:
        chrom_col = chrom_col[0]

    target_chr = chrom if chrom.startswith("chr") else f"chr{chrom}"
    df_chr = df[df[chrom_col].astype(str).str.replace("^chr", "chr", regex=True) == target_chr].copy()
    if df_chr.empty:
        target_chr_nochr = chrom.replace("chr", "")
        df_chr = df[df[chrom_col].astype(str) == target_chr_nochr].copy()

    n_bins = len(bins_df)
    track = np.full(n_bins, np.nan, dtype=float)
    if df_chr.empty:
        logger.warning(f"No phasing track records for {chrom} in {phasing_file}")
        return track

    # Find value column
    if col_name and col_name in df_chr.columns:
        val_col = col_name
    else:
        candidates = [
            "GC_gene_density_activity",
            "GC_fraction",
            "gene_tss_count",
            "activity",
            "score",
            "value",
        ]
        val_col = next((c for c in candidates if c in df_chr.columns), df_chr.columns[-1])

    # Overlap with bins
    start_col = df_chr.columns[1]
    end_col = df_chr.columns[2]
    starts = bins_df["start"].to_numpy()
    ends = bins_df["end"].to_numpy()
    mids = (starts + ends) / 2.0

    seg_starts = df_chr[start_col].to_numpy(dtype=float)
    seg_ends = df_chr[end_col].to_numpy(dtype=float)
    seg_vals = pd.to_numeric(df_chr[val_col], errors="coerce").to_numpy(dtype=float)

    for s_start, s_end, s_val in zip(seg_starts, seg_ends, seg_vals):
        mask = (mids >= s_start) & (mids < s_end)
        track[mask] = s_val

    return track


def run_cnv_latent_ssm(
    cooler_path: str,
    chrom: str,
    resolution: Optional[int] = 50000,
    cnv_mode: str = "external",
    cnv_file: Optional[str] = None,
    cnv_value_type: str = "copy_number",
    cnv_effect_scale: str = "log2",
    ploidy: float = 2.0,
    min_cn_threshold: float = 0.2,
    mask_deletions: bool = True,
    phasing_file: Optional[str] = None,
    phasing_col: Optional[str] = None,
    n_pcs: int = 10,
    model_type: str = "ar1",
    max_iter: int = 400,
    tol: float = 1e-5,
    loading_init: str = "pc1",
    freeze_cnv_effect: bool = False,
    sv_file: Optional[str] = None,
    sv_edge_strength: float = 0.0,
    sv_decouple_endpoints: bool = False,
    sv_cnv_weighted: bool = False,
    contact_graph_k: int = 0,
    contact_graph_min_distance_bins: int = 40,
    contact_graph_min_correlation: float = 0.3,
    contact_graph_mutual: bool = True,
    contact_graph_signed: bool = False,
    contact_graph_strength: float = 0.0,
    sv_distance_oe: bool = False,
    is_microc: bool = True,
    balance: bool = False,
    decouple_breakpoints: bool = True,
    breakpoint_min_gap_bins: int = 3,
    decouple_cnv_breakpoints: bool = False,
    cnv_breakpoint_threshold: float = 0.4,
    verbose: bool = True,
) -> Tuple[SSMResults, dict]:
    """Run full end-to-end CNV-aware SSM Compartment Calling pipeline.

    Parameters
    ----------
    cooler_path : str
        Path to .cool or .mcool file.
    chrom : str
        Chromosome name (e.g. 'chr8').
    resolution : int
        Resolution in bp.
    cnv_mode : str
        'external', 'infer', or 'none'.
    cnv_file : str, optional
        Path to external CNV bed/tsv file.
    cnv_effect_scale : str
        Encode the CNV observation covariate as log2(CN/P) or CN/P - 1.
    ploidy : float
        Sample baseline ploidy (default 2.0).
    min_cn_threshold : float
        Threshold for homozygous deletion masking (default 0.2).
    mask_deletions : bool
        Whether to exclude homozygous deletion regions from compartment calls (default True).
    phasing_file : str, optional
        Path to gene density or GC track used for post-fit sign alignment and,
        only when explicitly selected, phasing-based loading initialization.
    phasing_col : str, optional
        Column name in phasing_file to use.
    n_pcs : int
        Number of principal components for observation vector y_i (default 10).
    model_type : str
        'ar1', 'local_linear_trend', or 'two_scale' (global plus local artifact).
    max_iter : int
        Max EM iterations.
    tol : float
        EM convergence tolerance.
    loading_init : str
        Initialization for compartment loadings: ``pc1``, equal-norm
        ``uniform``, or correlations with the supplied ``phasing`` track.
    freeze_cnv_effect : bool
        Estimate PC-wise CN effects before EM and keep them fixed while fitting
        the latent compartment state.
    sv_file : str, optional
        EagleC/EagleC2 SV calls. Both endpoints are treated as AR transition
        boundaries so rearrangement-driven shifts do not propagate linearly.
    sv_edge_strength : float
        Precision multiplier for transitions connecting paired SV endpoints.
    sv_decouple_endpoints : bool
        Also cut ordinary chromosome-chain transitions at SV endpoints.
    sv_cnv_weighted : bool
        Weight paired SV transitions by CNVkit-derived endpoint dosage.
    contact_graph_k : int
        Number of distal contact-similarity neighbors per valid bin; zero disables.
    contact_graph_min_correlation : float
        Minimum O/E profile Pearson correlation for a graph edge.
    contact_graph_mutual : bool
        Require both endpoints to select one another among their top-k neighbors.
    contact_graph_signed : bool
        Use absolute-correlation neighbors and opposite-state constraints for
        negatively correlated profiles.
    sv_distance_oe : bool
        Recompute O/E expected using reference/SV single-edge distance mixtures.
    is_microc : bool
        True if Micro-C (no restriction enzyme bias term in CNV inference).
    balance : bool
        Whether to use balanced contacts. Default False (raw counts).
    decouple_breakpoints : bool
        Whether to decouple Markov transitions across structural breakpoints, gaps, and deletions (default True).
    breakpoint_min_gap_bins : int
        Minimum consecutive invalid bins required to restart the state chain.
        Shorter gaps are bridged (default 3).
    decouple_cnv_breakpoints : bool
        Restart the state chain at CNV jumps. Off by default because the CNV
        observation term already models copy-number changes.
    verbose : bool
        Whether to print logging progress.

    Returns
    -------
    results : SSMResults
        Estimated latent state trajectory, uncertainty, discrete calls, and parameters.
    extra_data : dict
        Additional matrices (O/E, correlation, observation PCA vectors, raw CNV, etc.)
    """
    logger.info(f"Loading contact matrix for {chrom} at {resolution}bp from {cooler_path}...")
    raw_matrix, bins_df = extract_chrom_matrix_and_bins(
        cooler_path=cooler_path,
        chrom=chrom,
        resolution=resolution,
        balance=balance,
    )
    n_bins = len(bins_df)
    logger.info(f"Extracted matrix shape: {raw_matrix.shape} ({n_bins} bins)")

    # 1. Identify valid unmasked bins (matching cooltools / cooler balancing criteria)
    valid_mask = identify_valid_bins(
        raw_matrix,
        bins_df=bins_df,
        require_balancing_weight=True,
    )

    # 2. Phasing track loading
    phasing_track = None
    if phasing_file is not None:
        logger.info(f"Loading phasing track from {phasing_file}...")
        phasing_track = load_phasing_track(
            phasing_file=phasing_file,
            chrom=chrom,
            bins_df=bins_df,
            col_name=phasing_col,
        )

    # 3. CNV deviation track c_i and homozygous deletion detection
    logger.info(f"Deriving CNV deviation track with mode='{cnv_mode}', ploidy={ploidy} (min_cn={min_cn_threshold})...")
    c_i, cn_raw, is_deleted, cnv_meta = get_cnv_track(
        mode=cnv_mode,
        chrom=chrom,
        bins_df=bins_df,
        external_cnv_file=cnv_file,
        ploidy=ploidy,
        min_cn_threshold=min_cn_threshold,
        value_type=cnv_value_type,
        effect_scale=cnv_effect_scale,
    )

    if mask_deletions:
        n_del = np.sum(is_deleted)
        if n_del > 0:
            logger.info(f"Masking {n_del} bins in homozygous/extreme deletion regions (CN <= {min_cn_threshold}) from compartment calling.")
            valid_mask = valid_mask & (~is_deleted)

    logger.info(f"Valid unmasked bins: {np.sum(valid_mask)} / {n_bins} ({np.mean(valid_mask)*100:.1f}%)")

    # 4. Distance normalization: Observed / Expected
    logger.info("Computing distance-dependent Observed/Expected (O/E) matrix...")
    sv_distance_edges = []
    if sv_distance_oe:
        if sv_file is None:
            raise ValueError("sv_distance_oe requires sv_file")
        cn_relative = cn_raw if cnv_value_type == "log2_ratio" else cn_raw / ploidy
        sv_distance_edges = load_sv_distance_edges(sv_file, chrom, bins_df, cn_relative)
        logger.info("Using %d confidence/CN-weighted single-SV distance edges for O/E.",
                    len(sv_distance_edges))
        oe_matrix = compute_sv_distance_mixture_oe(
            raw_matrix, valid_mask=valid_mask, sv_edges=sv_distance_edges,
        )
    else:
        oe_matrix = compute_observed_over_expected(raw_matrix, valid_mask=valid_mask)

    # 5. Whole-chromosome interaction profile correlation
    logger.info("Computing interaction profile Pearson correlation matrix S...")
    corr_matrix = compute_interaction_profile_correlation(oe_matrix, valid_mask=valid_mask)

    contact_graph_edges = []
    if contact_graph_k > 0 and contact_graph_strength > 0:
        contact_graph_edges = build_contact_knn_graph(
            corr_matrix, valid_mask, k=contact_graph_k,
            min_distance_bins=contact_graph_min_distance_bins,
            min_correlation=contact_graph_min_correlation,
            mutual=contact_graph_mutual,
            signed=contact_graph_signed,
        )
        logger.info(
            "Built distal contact graph with %d edges (k=%d, min distance=%d bins, "
            "min |r|/r=%.3f, mutual=%s, signed=%s).",
            len(contact_graph_edges), contact_graph_k, contact_graph_min_distance_bins,
            contact_graph_min_correlation, contact_graph_mutual, contact_graph_signed,
        )

    # 6. Observation feature extraction via PCA
    logger.info(f"Compressing correlation matrix into top {n_pcs} observation PCs...")
    y_obs, var_explained, pca_obj = extract_pca_observation_features(
        correlation_matrix=corr_matrix,
        valid_mask=valid_mask,
        n_components=n_pcs,
    )

    # 7. Identify structural breakpoints & transition boundaries
    breakpoints = None
    if decouple_breakpoints:
        breakpoints = identify_transition_breakpoints(
            bins_df=bins_df,
            valid_mask=valid_mask,
            cnv_deviation=c_i,
            min_gap_bins=breakpoint_min_gap_bins,
            decouple_cnv_jumps=decouple_cnv_breakpoints,
            cnv_jump_threshold=cnv_breakpoint_threshold,
        )
        logger.info(f"Identified {np.sum(breakpoints)} structural transition breakpoints for Markov decoupling.")
    sv_edges = []
    sv_edge_table = pd.DataFrame()
    if sv_file is not None:
        sv_breakpoints, sv_edges = load_sv_endpoint_breakpoints(
            sv_file=sv_file, chrom=chrom, bins_df=bins_df,
        )
        if sv_cnv_weighted:
            cn_relative = cn_raw if cnv_value_type == "log2_ratio" else cn_raw / ploidy
            sv_edges, sv_edge_table = weight_sv_edges_by_cnv(sv_edges, cn_relative)
            sv_edge_table["chrom1"] = chrom
            sv_edge_table["pos1"] = [int(bins_df.iloc[i]["start"]) for i in sv_edge_table["bin1"]]
            sv_edge_table["chrom2"] = chrom
            sv_edge_table["pos2"] = [int(bins_df.iloc[j]["start"]) for j in sv_edge_table["bin2"]]
            logger.info(
                "CNV-weighted SV transitions: relative weights %.3f to %.3f.",
                sv_edge_table["relative_edge_weight"].min(),
                sv_edge_table["relative_edge_weight"].max(),
            )
        if sv_decouple_endpoints:
            if breakpoints is None:
                breakpoints = sv_breakpoints
                if len(breakpoints):
                    breakpoints[0] = True
            else:
                breakpoints = breakpoints | sv_breakpoints
        logger.info(
            "Loaded %d intra-chromosomal SV edges; paired graph transitions enabled%s.",
            len(sv_edges),
            " and endpoints decoupled" if sv_decouple_endpoints else " without cutting linear AR",
        )

    # 8. Fit CNV-aware State Space Model
    logger.info(f"Fitting CNV-aware SSM (model_type='{model_type}', n_pcs={n_pcs}, max_iter={max_iter})...")
    ssm = CNVAwareSSM(
        model_type=model_type,
        n_pcs=n_pcs,
        max_iter=max_iter,
        tol=tol,
        loading_init=loading_init,
        freeze_cnv_effect=freeze_cnv_effect,
        sv_edge_strength=sv_edge_strength,
        contact_graph_strength=contact_graph_strength,
        verbose=verbose,
    )
    fit_dict = ssm.fit(
        y=y_obs,
        c=c_i,
        valid_mask=valid_mask,
        prior_phasing_track=phasing_track,
        breakpoints=breakpoints,
        sv_edges=sv_edges,
        contact_graph_edges=contact_graph_edges,
    )

    state_score = fit_dict["state_score"]
    posterior_sd = fit_dict["posterior_sd"]
    p_active = fit_dict["p_active"]
    discrete_state = fit_dict["discrete_state"]
    confidence = fit_dict["confidence"]
    state_slope = fit_dict.get("state_slope")
    local_artifact_score = fit_dict.get("local_artifact_score")

    # Strictly mask out deleted regions so they have NO compartment call
    if mask_deletions and np.any(is_deleted):
        state_score[is_deleted] = np.nan
        posterior_sd[is_deleted] = np.nan
        p_active[is_deleted] = np.nan
        discrete_state[is_deleted] = "NA"
        confidence[is_deleted] = "NA"
        if state_slope is not None:
            state_slope[is_deleted] = np.nan
        if local_artifact_score is not None:
            local_artifact_score[is_deleted] = np.nan

    # Build results dataclass
    results = SSMResults(
        chrom=chrom,
        bins=bins_df,
        state_score=state_score,
        posterior_sd=posterior_sd,
        state_slope=state_slope,
        local_artifact_score=local_artifact_score,
        p_active=p_active,
        discrete_state=discrete_state,
        confidence=confidence,
        cnv_deviation=c_i,
        cnv_raw=cn_raw,
        model_type=model_type,
        F=fit_dict["F"],
        H=fit_dict["H"],
        G=fit_dict["G"],
        Q=fit_dict["Q"],
        R=fit_dict["R"],
        log_likelihood=fit_dict["log_likelihood"],
        iterations=fit_dict["iterations"],
        converged=fit_dict["converged"],
        loglik_history=fit_dict["loglik_history"],
        explained_variance_ratio=var_explained,
    )

    extra_data = {
        "raw_matrix": raw_matrix,
        "oe_matrix": oe_matrix,
        "corr_matrix": corr_matrix,
        "y_obs": y_obs,
        "valid_mask": valid_mask,
        "is_deleted": is_deleted,
        "phasing_track": phasing_track,
        "cnv_meta": cnv_meta,
        "pca_obj": pca_obj,
        "sv_edges": sv_edges,
        "sv_edge_table": sv_edge_table,
        "contact_graph_edges": contact_graph_edges,
        "sv_distance_edges": sv_distance_edges,
    }

    return results, extra_data
