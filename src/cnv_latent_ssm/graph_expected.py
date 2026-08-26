"""Copy-flow additive expected-contact model for rearranged genomes.

This module deliberately consumes an oriented derivative-walk decomposition.
It does not infer junction dosage from endpoint segment CN and does not collapse
multiple derivative alleles to one shortest path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass
class GraphExpectedResult:
    expected: np.ndarray
    cis_expected: np.ndarray
    external_expected: np.ndarray
    cis_copy_pairs: np.ndarray
    external_copy_pairs: np.ndarray
    total_copy_pairs: np.ndarray
    external_level: float
    external_beta: float


def estimate_native_decay(matrix: np.ndarray, valid_mask: np.ndarray,
                          background_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate native P(d) only from caller-supplied background pixels."""
    n = matrix.shape[0]
    valid = np.outer(valid_mask, valid_mask)
    if background_mask is not None:
        valid &= np.asarray(background_mask, dtype=bool)
    curve = np.full(n, np.nan, dtype=float)
    for distance in range(n):
        values = np.diag(matrix, k=distance)
        keep = np.diag(valid, k=distance)
        positive = keep & np.isfinite(values) & (values >= 0)
        if positive.any():
            curve[distance] = np.median(values[positive])
    supported = np.flatnonzero(np.isfinite(curve) & (curve > 0))
    if not len(supported):
        raise ValueError("no supported native-distance background pixels")
    curve = np.interp(np.arange(n), supported, curve[supported])
    # A native decay curve must not increase with genomic distance.
    return np.minimum.accumulate(curve)


def _walk_bin_coordinates(bins: pd.DataFrame, walk: pd.DataFrame):
    centers = (bins["start"].to_numpy(float) + bins["end"].to_numpy(float)) / 2
    ids, positions = [], []
    offset = 0.0
    for node in walk.sort_values("order").itertuples():
        hit = np.flatnonzero((centers >= node.start) & (centers < node.end))
        if str(node.strand) == "+":
            position = offset + centers[hit] - float(node.start)
        elif str(node.strand) == "-":
            position = offset + float(node.end) - centers[hit]
        else:
            raise ValueError("walk strand must be '+' or '-'")
        ids.append(hit)
        positions.append(position)
        offset += float(node.end - node.start)
    if not ids:
        return np.array([], dtype=int), np.array([]), offset
    return np.concatenate(ids), np.concatenate(positions), offset


def accumulate_oriented_walks(bins: pd.DataFrame, walk_nodes: pd.DataFrame,
                              native_decay: np.ndarray, resolution: int,
                              ploidy: float = 2.0):
    """Sum copy-specific P(d) over all oriented derivative walks.

    Each walk contributes independently. A deletion junction therefore gives
    its direct allele a one-bin graph distance, while reference and alternative
    alleles retain their own distances. Repeated traversal remains repeated CN.
    """
    required = {"walk_id", "walk_cn", "circular", "order",
                "start", "end", "strand"}
    missing = required.difference(walk_nodes.columns)
    if missing:
        raise ValueError(f"walk_nodes missing columns: {sorted(missing)}")
    n = len(bins)
    cis_expected = np.zeros((n, n), dtype=float)
    cis_pairs = np.zeros((n, n), dtype=float)
    for _, walk in walk_nodes.groupby("walk_id", sort=False):
        ids, position, length = _walk_bin_coordinates(bins, walk)
        if not len(ids):
            continue
        distance = np.abs(position[:, None] - position[None, :])
        if bool(walk["circular"].iloc[0]) and length > 0:
            distance = np.minimum(distance, length - distance)
        distance_bins = np.clip(np.rint(distance / resolution).astype(int),
                                0, len(native_decay) - 1)
        # Two distinct reference bins cannot occupy the same physical locus on
        # a derivative molecule.  Sub-bin breakpoint offsets may round to zero,
        # but using P(0) there would import the much larger self-contact term.
        off_diagonal = ids[:, None] != ids[None, :]
        distance_bins[off_diagonal] = np.maximum(
            distance_bins[off_diagonal], 1)
        copy = float(walk["walk_cn"].iloc[0]) / ploidy
        rows = np.broadcast_to(ids[:, None], distance_bins.shape).ravel()
        cols = np.broadcast_to(ids[None, :], distance_bins.shape).ravel()
        np.add.at(cis_expected, (rows, cols),
                  (copy * native_decay[distance_bins]).ravel())
        np.add.at(cis_pairs, (rows, cols), copy)
    return cis_expected, cis_pairs


def fit_external_exposure(matrix: np.ndarray, external_pairs: np.ndarray,
                          valid_mask: np.ndarray, unconnected_mask: np.ndarray,
                          beta_bounds=(0.25, 1.5)):
    """Fit P_ext and CN exponent from graph-unconnected copy pairs."""
    use = (np.outer(valid_mask, valid_mask) & unconnected_mask
           & (external_pairs > 0))
    upper = np.triu(use, k=1)
    observed = matrix[upper].astype(float)
    dosage = external_pairs[upper].astype(float)
    if not len(observed):
        raise ValueError("no graph-unconnected pairs available for P_ext")

    def objective(beta):
        powered = dosage ** beta
        level = observed.sum() / max(powered.sum(), 1e-12)
        mean = np.maximum(level * powered, 1e-12)
        return float(np.sum(mean - observed * np.log(mean)))

    fit = minimize_scalar(objective, bounds=beta_bounds, method="bounded")
    beta = float(fit.x)
    powered = dosage ** beta
    level = float(observed.sum() / max(powered.sum(), 1e-12))
    return level, beta


def compute_copy_flow_additive_oe(
    matrix: np.ndarray,
    bins: pd.DataFrame,
    valid_mask: np.ndarray,
    segment_cn: np.ndarray,
    walk_nodes: pd.DataFrame,
    native_decay: np.ndarray,
    resolution: int,
    external_level: Optional[float] = None,
    external_beta: float = 1.0,
    external_fit_mask: Optional[np.ndarray] = None,
    ploidy: float = 2.0,
):
    """Compute O/E from additive same-molecule and external copy pools."""
    cis_expected, raw_cis_pairs = accumulate_oriented_walks(
        bins, walk_nodes, native_decay, resolution, ploidy)
    dosage = np.maximum(np.asarray(segment_cn, float) / ploidy, 0)
    total_pairs = np.outer(dosage, dosage)
    # raw_cis_pairs is a relative molecule multiplicity M/P, whereas
    # total_pairs is a relative copy-pair pool CN_i*CN_j/P^2.  Cap M/P by the
    # copies available at either endpoint; do not compare it to total_pairs.
    max_cis_rel = np.minimum(dosage[:, None], dosage[None, :])
    cis_rel = np.minimum(raw_cis_pairs, max_cis_rel)
    scale = np.divide(cis_rel, raw_cis_pairs, out=np.zeros_like(cis_rel),
                      where=raw_cis_pairs > 0)
    cis_expected *= scale
    # Convert M/P to the same pair units as CN_i*CN_j/P^2 before subtraction.
    cis_pairs = cis_rel / ploidy
    external_pairs = np.maximum(total_pairs - cis_pairs, 0)
    if external_level is None:
        if external_fit_mask is None:
            external_fit_mask = raw_cis_pairs <= 0
        external_level, external_beta = fit_external_exposure(
            matrix, external_pairs, valid_mask, external_fit_mask)
    external_expected = external_level * external_pairs ** external_beta
    expected = cis_expected + external_expected
    usable = np.outer(valid_mask, valid_mask) & (expected > 0)
    oe = np.zeros_like(matrix, dtype=float)
    oe[usable] = matrix[usable] / expected[usable]
    return oe, GraphExpectedResult(
        expected, cis_expected, external_expected, cis_pairs,
        external_pairs, total_pairs, float(external_level),
        float(external_beta))


def flow_residuals(segment_cn: np.ndarray, incident_junction_cn: np.ndarray,
                   source_cn: np.ndarray) -> np.ndarray:
    """Return segment-side flow residuals; source is explicit, not an SV edge."""
    target = np.repeat(np.asarray(segment_cn, float), 2)
    incident = np.asarray(incident_junction_cn, float).reshape(-1)
    source = np.asarray(source_cn, float).reshape(-1)
    if not (len(target) == len(incident) == len(source)):
        raise ValueError("flow arrays must describe both sides of every segment")
    return incident + source - target
