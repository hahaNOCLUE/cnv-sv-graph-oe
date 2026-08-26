"""CNV Module: External loading, deletion identification, and deviation computation."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_external_cnv(
    cnv_file: str,
    chrom: str,
    bins_df: pd.DataFrame,
    ploidy: float = 2.0,
    min_cn_threshold: float = 0.2,
    col_cn: Optional[str] = None,
    value_type: str = "copy_number",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load external CNV intervals, identify deleted bins, and compute deviations.

    Parameters
    ----------
    cnv_file : str
        Path to external CNV bed/tsv file with columns: [chrom, start, end, CN/total_CN].
    chrom : str
        Target chromosome name (e.g. 'chr8' or '8').
    bins_df : pd.DataFrame
        Genomic bins with columns: ['chrom', 'start', 'end'].
    ploidy : float
        Sample baseline ploidy P (default 2.0).
    min_cn_threshold : float
        Threshold below which a region is treated as homozygous / extreme deletion (default 0.2).
    col_cn : str, optional
        Name of copy number column if file has headers.

    Returns
    -------
    c_i : np.ndarray
        Copy number deviation: c_i = log2(CN_i / P).
    cn_i : np.ndarray
        Estimated absolute copy number for each bin.
    is_deleted : np.ndarray
        Boolean mask indicating deleted regions (CN <= min_cn_threshold).
    """
    has_header = cnv_file.lower().endswith((".cns", ".cnr"))
    df = pd.read_csv(cnv_file, sep=r"\s+", header=0 if has_header else None, comment="#")
    if df.shape[1] < 4:
        raise ValueError(f"External CNV file {cnv_file} must have at least 4 columns (chrom, start, end, CN)")

    # Check if first row is header
    if has_header or (isinstance(df.iloc[0, 1], str) and not df.iloc[0, 1].isdigit()):
        df = pd.read_csv(cnv_file, sep=r"\s+", comment="#")
        chrom_col, start_col, end_col = df.columns[0], df.columns[1], df.columns[2]
        if col_cn is not None:
            cn_col = col_cn
        elif value_type == "log2_ratio" and "log2" in df.columns:
            cn_col = "log2"
        else:
            cn_col = df.columns[3]
    else:
        chrom_col, start_col, end_col, cn_col = 0, 1, 2, 3

    # Normalize chromosome name
    target_chr = chrom if chrom.startswith("chr") else f"chr{chrom}"
    df_chr = df[
        df[chrom_col].astype(str).str.replace("^chr", "chr", regex=True) == target_chr
    ].copy()
    if df_chr.empty:
        target_chr_nochr = chrom.replace("chr", "")
        df_chr = df[df[chrom_col].astype(str) == target_chr_nochr].copy()

    n_bins = len(bins_df)
    default_value = 0.0 if value_type == "log2_ratio" else ploidy
    cn_profile = np.full(n_bins, default_value, dtype=float)

    if df_chr.empty:
        logger.warning(f"No CNV records found for chromosome {chrom} in {cnv_file}. Defaulting to ploidy={ploidy}.")
    else:
        # Overlap CNV segments with bins
        starts = bins_df["start"].to_numpy()
        ends = bins_df["end"].to_numpy()
        mids = (starts + ends) / 2.0

        seg_starts = df_chr[start_col].to_numpy(dtype=float)
        seg_ends = df_chr[end_col].to_numpy(dtype=float)
        seg_cns = df_chr[cn_col].to_numpy(dtype=float)

        for s_start, s_end, s_cn in zip(seg_starts, seg_ends, seg_cns):
            mask = (mids >= s_start) & (mids < s_end)
            cn_profile[mask] = s_cn

    if value_type == "log2_ratio":
        c_i = cn_profile.copy()
        relative_ratio = np.exp2(c_i)
        # Keep the historical deletion criterion (CN <= 0.2 at ploidy 2)
        # expressed as a relative-copy threshold, without altering c_i.
        relative_deletion_threshold = min_cn_threshold / ploidy
        is_deleted = relative_ratio <= relative_deletion_threshold
        cn_profile = relative_ratio
    else:
        is_deleted = cn_profile <= min_cn_threshold
        cn_clamped = np.maximum(cn_profile, 0.1)
        c_i = np.log2(cn_clamped / ploidy)
    return c_i, cn_profile, is_deleted


def get_cnv_track(
    mode: str,
    chrom: str,
    bins_df: pd.DataFrame,
    external_cnv_file: Optional[str] = None,
    ploidy: float = 2.0,
    min_cn_threshold: float = 0.2,
    value_type: str = "copy_number",
    effect_scale: str = "log2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Retrieve CNV deviation track c_i and deleted bins based on mode ('external' or 'none')."""
    mode = mode.lower()
    if effect_scale not in {"log2", "linear"}:
        raise ValueError("effect_scale must be 'log2' or 'linear'")
    meta = {
        "mode": mode,
        "ploidy": ploidy,
        "min_cn_threshold": min_cn_threshold,
        "effect_scale": effect_scale,
    }

    if mode == "none" or external_cnv_file is None:
        c_i = np.zeros(len(bins_df), dtype=float)
        cn_raw = np.full(len(bins_df), ploidy, dtype=float)
        is_deleted = np.zeros(len(bins_df), dtype=bool)
        return c_i, cn_raw, is_deleted, meta

    elif mode == "external":
        c_i, cn_raw, is_deleted = load_external_cnv(
            external_cnv_file,
            chrom=chrom,
            bins_df=bins_df,
            ploidy=ploidy,
            min_cn_threshold=min_cn_threshold,
            value_type=value_type,
        )
        if effect_scale == "linear":
            # cn_raw is CN/P for log2-ratio inputs and absolute CN otherwise.
            relative_cn = cn_raw if value_type == "log2_ratio" else cn_raw / ploidy
            c_i = relative_cn - 1.0
        return c_i, cn_raw, is_deleted, meta

    else:
        raise ValueError(f"Unknown CNV mode: {mode}. Choose from 'external' or 'none'.")
