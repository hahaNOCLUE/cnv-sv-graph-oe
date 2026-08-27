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


@dataclass
class GraphExpectedResult:
    expected: np.ndarray
    cis_expected: np.ndarray
    external_expected: np.ndarray
    cis_copy_pairs: np.ndarray
    external_copy_pairs: np.ndarray
    total_copy_pairs: np.ndarray
    collision_floor: float
    capture_visibility: np.ndarray


def _weighted_nonincreasing_isotonic(values: np.ndarray,
                                      weights: np.ndarray) -> np.ndarray:
    """Weighted PAVA fit constrained to be non-increasing."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index + 1, float(value), float(weight)])
        while len(blocks) >= 2 and blocks[-2][2] < blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[3] + right[3]
            mean = ((left[2] * left[3] + right[2] * right[3])
                    / max(total_weight, 1e-12))
            blocks.append([left[0], right[1], mean, total_weight])
    fitted = np.empty(len(values), dtype=float)
    for start, end, value, _ in blocks:
        fitted[start:end] = value
    return fitted


def estimate_native_decay(matrix: np.ndarray, valid_mask: np.ndarray,
                          background_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Estimate native P(d) using weighted isotonic regression on log counts."""
    n = matrix.shape[0]
    valid = np.outer(valid_mask, valid_mask)
    if background_mask is not None:
        valid &= np.asarray(background_mask, dtype=bool)
    curve = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=float)
    for distance in range(n):
        values = np.diag(matrix, k=distance)
        keep = np.diag(valid, k=distance)
        positive = keep & np.isfinite(values) & (values > 0)
        if positive.any():
            curve[distance] = np.median(values[positive])
            counts[distance] = positive.sum()
    supported = np.flatnonzero(np.isfinite(curve) & (curve > 0))
    if not len(supported):
        raise ValueError("no supported native-distance background pixels")
    fitted_log = _weighted_nonincreasing_isotonic(
        np.log(curve[supported]), counts[supported])
    # Interpolation occurs after fitting, so an isolated low diagonal cannot
    # permanently drag every more-distal expected value downward.
    return np.exp(np.interp(np.arange(n), supported, fitted_log))


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


def fit_collision_floor(matrix: np.ndarray, intermolecular_pairs: np.ndarray,
                        valid_mask: np.ndarray,
                        unconnected_mask: np.ndarray) -> float:
    """Fit constant B in B*N_inter using graph-unconnected pixels only."""
    use = (np.outer(valid_mask, valid_mask) & unconnected_mask
           & (intermolecular_pairs > 0))
    upper = np.triu(use, k=1)
    observed = matrix[upper].astype(float)
    dosage = intermolecular_pairs[upper].astype(float)
    if not len(observed):
        raise ValueError("no graph-unconnected pairs available for collision fit")
    return float(observed.sum() / max(dosage.sum(), 1e-12))


def compute_copy_flow_additive_oe(
    matrix: np.ndarray,
    bins: pd.DataFrame,
    valid_mask: np.ndarray,
    segment_cn: np.ndarray,
    walk_nodes: pd.DataFrame,
    native_decay: np.ndarray,
    resolution: int,
    collision_floor: Optional[float] = None,
    external_fit_mask: Optional[np.ndarray] = None,
    ploidy: float = 2.0,
    capture_visibility: Optional[np.ndarray] = None,
):
    """Compute O/E from physical copies, topology, distance and visibility.

    ``capture_visibility`` must be an externally specified technical track
    (for example mappability/MNase/GC recovery), never a free long-range bin
    effect learned from the contact matrix.
    """
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
    if capture_visibility is None:
        visibility = np.ones(len(dosage), dtype=float)
    else:
        visibility = np.asarray(capture_visibility, float)
        if visibility.shape != dosage.shape:
            raise ValueError("capture_visibility must have one value per bin")
        if np.any(~np.isfinite(visibility)) or np.any(visibility <= 0):
            raise ValueError("capture_visibility must be finite and positive")
    visibility_pair = np.outer(visibility, visibility)
    if collision_floor is None:
        if external_fit_mask is None:
            external_fit_mask = raw_cis_pairs <= 0
        visibility_corrected = matrix / visibility_pair
        collision_floor = fit_collision_floor(
            visibility_corrected, external_pairs, valid_mask,
            external_fit_mask)
    external_expected = collision_floor * external_pairs
    cis_expected *= visibility_pair
    external_expected *= visibility_pair
    expected = cis_expected + external_expected
    usable = np.outer(valid_mask, valid_mask) & (expected > 0)
    oe = np.zeros_like(matrix, dtype=float)
    oe[usable] = matrix[usable] / expected[usable]
    return oe, GraphExpectedResult(
        expected, cis_expected, external_expected, cis_pairs,
        external_pairs, total_pairs, float(collision_floor), visibility)


def flow_residuals(segment_cn: np.ndarray, incident_junction_cn: np.ndarray,
                   source_cn: np.ndarray) -> np.ndarray:
    """Return segment-side flow residuals; source is explicit, not an SV edge."""
    target = np.repeat(np.asarray(segment_cn, float), 2)
    incident = np.asarray(incident_junction_cn, float).reshape(-1)
    source = np.asarray(source_cn, float).reshape(-1)
    if not (len(target) == len(incident) == len(source)):
        raise ValueError("flow arrays must describe both sides of every segment")
    return incident + source - target
