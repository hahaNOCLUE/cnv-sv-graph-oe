"""Genome-wide trans-contact calibration for the external copy-pair model."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cooler
import numpy as np
import pandas as pd


def _cooler_uri(path: str, resolution: Optional[int]) -> str:
    if "::" in path or not str(path).endswith(".mcool"):
        return path
    if resolution is None:
        raise ValueError("resolution is required for an mcool trans fit")
    return f"{path}::/resolutions/{resolution}"


def _absolute_cn(cns: pd.DataFrame, bins: pd.DataFrame,
                 ploidy: float) -> np.ndarray:
    required = {"chromosome", "start", "end", "log2"}
    missing = required.difference(cns.columns)
    if missing:
        raise ValueError(f"CNVkit CNS missing columns: {sorted(missing)}")
    centers = (bins["start"].to_numpy() + bins["end"].to_numpy()) / 2
    chrom = bins["chrom"].astype(str).to_numpy()
    cn = np.full(len(bins), np.nan)
    for row in cns.itertuples(index=False):
        value = ploidy * 2.0 ** float(row.log2)
        hit = ((chrom == str(row.chromosome)) & (centers >= float(row.start))
               & (centers < float(row.end)))
        cn[hit] = value
    return cn


def fit_trans_external_from_cooler(
    cooler_path: str,
    resolution: Optional[int],
    cnvkit_cns: str,
    ploidy: float = 2.0,
    chromosomes: Optional[Sequence[str]] = None,
):
    """Fit constant collision floor ``B`` from raw trans contacts.

    All valid zero-valued trans pixels enter the Poisson exposure exactly via
    separable dosage sums.  Only nonzero sparse pixels need to be materialized.
    """
    clr = cooler.Cooler(_cooler_uri(cooler_path, resolution))
    bins = clr.bins()[:].reset_index(drop=True)
    cns = pd.read_csv(Path(cnvkit_cns), sep="\t")
    cn = _absolute_cn(cns, bins, ploidy)
    weights = (pd.to_numeric(bins.get("weight", pd.Series(np.ones(len(bins)))),
                             errors="coerce").to_numpy(float))
    valid = np.isfinite(weights) & (weights > 0) & np.isfinite(cn) & (cn > 0)
    dosage = cn / ploidy

    available = [str(c) for c in clr.chromnames]
    if chromosomes is None:
        def autosome_number(name: str):
            label = name[3:] if name.startswith("chr") else name
            return int(label) if label.isdigit() else None

        chromosomes = [c for c in available
                       if autosome_number(c) is not None
                       and 1 <= autosome_number(c) <= 22]
    chromosomes = [c for c in chromosomes if c in available]
    offsets = clr.offset
    chrom_data = {}
    for chrom in chromosomes:
        lo, hi = offsets(chrom), offsets(chrom) + len(clr.bins().fetch(chrom))
        chrom_data[chrom] = (valid[lo:hi], dosage[lo:hi])

    pair_rows = []
    observed_total = 0.0
    pair_terms = []
    selector = clr.matrix(balance=False, sparse=True)
    for left_index, left in enumerate(chromosomes):
        valid_left, dosage_left = chrom_data[left]
        for right in chromosomes[left_index + 1:]:
            valid_right, dosage_right = chrom_data[right]
            if not valid_left.any() or not valid_right.any():
                continue
            matrix = selector.fetch(left, right).tocoo()
            keep = (valid_left[matrix.row] & valid_right[matrix.col]
                    & np.isfinite(matrix.data) & (matrix.data > 0))
            values = matrix.data[keep].astype(float)
            pair_dosage = (dosage_left[matrix.row[keep]]
                           * dosage_right[matrix.col[keep]])
            observed = float(values.sum())
            observed_total += observed
            pair_terms.append((dosage_left[valid_left], dosage_right[valid_right]))
            pair_rows.append({
                "chrom1": left, "chrom2": right,
                "valid_pair_count": int(valid_left.sum() * valid_right.sum()),
                "nonzero_pixel_count": int(keep.sum()),
                "observed_count": observed,
            })
    if observed_total <= 0 or not pair_terms:
        raise ValueError("no valid trans contacts available for external fit")

    total_exposure = float(sum(left.sum() * right.sum()
                               for left, right in pair_terms))
    level = float(observed_total / max(total_exposure, 1e-12))
    diagnostics = pd.DataFrame(pair_rows)
    diagnostics["expected_count"] = [
        level * left.sum() * right.sum()
        for left, right in pair_terms
    ]
    diagnostics["observed_expected_ratio"] = (
        diagnostics.observed_count / diagnostics.expected_count)
    return level, diagnostics
