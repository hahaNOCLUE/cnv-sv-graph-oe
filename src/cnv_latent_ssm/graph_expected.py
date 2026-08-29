"""Copy-flow additive expected-contact model for rearranged genomes.

This module deliberately consumes an oriented derivative-walk decomposition.
It does not infer junction dosage from endpoint segment CN and does not collapse
multiple derivative alleles to one shortest path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
    min_graph_distance_bins: np.ndarray
    visibility_band_cis_expected: np.ndarray


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
    """Estimate zero-inclusive baseline total contact rate by distance."""
    n = matrix.shape[0]
    valid = np.outer(valid_mask, valid_mask)
    if background_mask is not None:
        valid &= np.asarray(background_mask, dtype=bool)
    curve = np.full(n, np.nan, dtype=float)
    counts = np.zeros(n, dtype=float)
    for distance in range(n):
        values = np.diag(matrix, k=distance)
        keep = np.diag(valid, k=distance)
        eligible = keep & np.isfinite(values) & (values >= 0)
        if eligible.any():
            curve[distance] = np.mean(values[eligible])
            counts[distance] = eligible.sum()
    supported = np.flatnonzero(np.isfinite(curve) & (curve >= 0))
    if not len(supported):
        raise ValueError("no supported native-distance background pixels")
    positive_rate = curve[supported][curve[supported] > 0]
    epsilon = (max(float(positive_rate.min()) * 0.01, 1e-12)
               if len(positive_rate) else 1e-12)
    fitted_log = _weighted_nonincreasing_isotonic(
        np.log(curve[supported] + epsilon), counts[supported])
    # Interpolation occurs after fitting, so an isolated low diagonal cannot
    # permanently drag every more-distal expected value downward.
    return np.maximum(
        np.exp(np.interp(np.arange(n), supported, fitted_log)) - epsilon, 0)


def subtract_baseline_collision(baseline_decay: np.ndarray,
                                collision_floor: float,
                                ploidy: float = 2.0) -> np.ndarray:
    """Recover same-molecule P(d) from baseline total contact decay."""
    if ploidy <= 0:
        raise ValueError("ploidy must be positive")
    baseline_external_pairs = 1.0 - 1.0 / ploidy
    return np.maximum(np.asarray(baseline_decay, float)
                      - collision_floor * baseline_external_pairs, 0)


def fit_joint_same_decay(matrix: np.ndarray, valid_mask: np.ndarray,
                         background_mask: Optional[np.ndarray],
                         trans_collision_floor: float, ploidy: float = 2.0,
                         n_knots: int = 20, prior_log_sd: float = 0.7):
    """Jointly fit positive monotone P_same(d) and an intra collision floor.

    The likelihood uses all eligible diagonal pixels, including zero counts.
    A low-dimensional piecewise-linear curve in log-distance/log-rate space
    prevents the collision floor from being identified by clipping a flexible
    distance curve. ``B_trans`` supplies only a weak log-normal anchor.
    """
    if ploidy < 1 or trans_collision_floor <= 0:
        raise ValueError("joint decay fit requires ploidy >= 1 and B_trans > 0")
    n = matrix.shape[0]
    eligible_grid = np.outer(valid_mask, valid_mask)
    if background_mask is not None:
        eligible_grid &= np.asarray(background_mask, dtype=bool)
    distances, totals, counts = [], [], []
    for distance in range(1, n):
        values = np.diag(matrix, k=distance)
        keep = np.diag(eligible_grid, k=distance)
        use = keep & np.isfinite(values) & (values >= 0)
        if use.any():
            distances.append(distance)
            totals.append(float(values[use].sum()))
            counts.append(int(use.sum()))
    distances = np.asarray(distances, float)
    totals = np.asarray(totals, float)
    counts = np.asarray(counts, float)
    if len(distances) < 2:
        raise ValueError("insufficient baseline distances for joint decay fit")
    knot_x = np.linspace(np.log(distances.min()), np.log(distances.max()),
                         min(n_knots, len(distances)))
    log_distance = np.log(distances)
    rate = totals / counts
    floor_fraction = 1.0 - 1.0 / ploidy
    initial_b = trans_collision_floor
    initial_rate = np.maximum(rate - initial_b * floor_fraction, 1e-6)
    initial_log_p = np.interp(knot_x, log_distance, np.log(initial_rate))
    initial_log_p = _weighted_nonincreasing_isotonic(
        initial_log_p, np.ones_like(initial_log_p))
    x0 = np.r_[initial_log_p, np.log(initial_b)]
    objective_scale = max(float(counts.sum()), 1.0)

    def objective(parameters):
        log_p = np.interp(log_distance, knot_x, parameters[:-1])
        b_intra = np.exp(parameters[-1])
        mu = np.exp(log_p) + floor_fraction * b_intra
        nll = np.sum(counts * mu - totals * np.log(np.maximum(mu, 1e-12)))
        prior = .5 * ((parameters[-1] - np.log(trans_collision_floor)) /
                      prior_log_sd) ** 2
        # Scaling does not change the optimum, but avoids SLSQP line-search
        # failures on chromosome-scale count totals.
        return (nll + prior) / objective_scale

    constraints = [
        {"type": "ineq", "fun": lambda x, i=i: x[i] - x[i + 1]}
        for i in range(len(knot_x) - 1)
    ]
    fit = minimize(objective, x0, method="SLSQP", constraints=constraints,
                   bounds=[(-30, 30)] * len(knot_x) + [(-30, 10)],
                   options={"maxiter": 3000, "ftol": 1e-9})
    if not fit.success:
        raise RuntimeError(f"joint same-decay fit failed: {fit.message}")
    all_distance = np.arange(n, dtype=float)
    all_log_distance = np.log(np.maximum(all_distance, 1))
    same_decay = np.exp(np.interp(all_log_distance, knot_x, fit.x[:-1]))
    same_decay[0] = same_decay[1]
    return same_decay, float(np.exp(fit.x[-1]))


def aggregate_observed_expected(observed: np.ndarray, expected: np.ndarray,
                                valid_mask: np.ndarray, factor: int):
    """Coarsen counts and expected separately, then return sum(O)/sum(E)."""
    if factor < 1:
        raise ValueError("factor must be positive")
    n = len(valid_mask)
    coarse_n = int(np.ceil(n / factor))
    size = coarse_n * factor
    valid = np.outer(valid_mask, valid_mask)
    observed_pad = np.zeros((size, size), dtype=float)
    expected_pad = np.zeros((size, size), dtype=float)
    observed_pad[:n, :n] = np.where(valid, observed, 0)
    expected_pad[:n, :n] = np.where(valid, expected, 0)
    axes = (1, 3)
    observed_sum = observed_pad.reshape(
        coarse_n, factor, coarse_n, factor).sum(axis=axes)
    expected_sum = expected_pad.reshape(
        coarse_n, factor, coarse_n, factor).sum(axis=axes)
    oe = np.divide(observed_sum, expected_sum, out=np.zeros_like(observed_sum),
                   where=expected_sum > 0)
    coarse_valid = np.array([
        valid_mask[i * factor:min((i + 1) * factor, n)].any()
        for i in range(coarse_n)
    ])
    return observed_sum, expected_sum, oe, coarse_valid


def _walk_bin_coordinates(bins: pd.DataFrame, walk: pd.DataFrame):
    centers = (bins["start"].to_numpy(float) + bins["end"].to_numpy(float)) / 2
    bin_chrom = (bins["chrom"].astype(str).to_numpy()
                 if "chrom" in bins else None)
    ids, positions = [], []
    offset = 0.0
    for node in walk.sort_values("order").itertuples():
        in_node = (centers >= node.start) & (centers < node.end)
        if bin_chrom is not None and hasattr(node, "chrom"):
            in_node &= bin_chrom == str(node.chrom)
        hit = np.flatnonzero(in_node)
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
                              ploidy: float = 2.0,
                              visibility_distance_bp=(500_000, 5_000_000),
                              return_visibility_band: bool = False):
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
    visibility_band_expected = np.zeros((n, n), dtype=float)
    cis_pairs = np.zeros((n, n), dtype=float)
    min_distance = np.full((n, n), np.inf, dtype=float)
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
        in_visibility_band = ((distance >= visibility_distance_bp[0])
                              & (distance <= visibility_distance_bp[1]))
        np.add.at(visibility_band_expected, (rows, cols),
                  (copy * native_decay[distance_bins]
                   * in_visibility_band).ravel())
        np.add.at(cis_pairs, (rows, cols), copy)
        np.minimum.at(min_distance, (rows, cols), distance_bins.ravel())
    if return_visibility_band:
        return (cis_expected, cis_pairs, min_distance,
                visibility_band_expected)
    return cis_expected, cis_pairs, min_distance


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
    (cis_expected, raw_cis_pairs, min_graph_distance,
     visibility_band_cis_expected) = accumulate_oriented_walks(
        bins, walk_nodes, native_decay, resolution, ploidy,
        return_visibility_band=True)
    dosage = np.maximum(np.asarray(segment_cn, float) / ploidy, 0)
    total_pairs = np.outer(dosage, dosage)
    # raw_cis_pairs is occurrence-pair multiplicity M_occ/P. Repeated copies
    # of a locus on one derivative molecule are distinct physical pairs, so
    # min(CN_i, CN_j) is not a valid cap. Convert to total-pair units and only
    # check nominal copy-pair conservation. With fractional/subclonal CN,
    # CN_i*CN_j is a product of marginal dosages and is not a strict joint
    # upper bound, so violations are diagnostic and must not erase walk mass.
    max_cis_rel = total_pairs * ploidy
    tolerance = 1e-8 + 1e-6 * max_cis_rel
    excess = raw_cis_pairs > max_cis_rel + tolerance
    if np.any(excess):
        warnings.warn(
            f"same-molecule occurrence mass exceeds total copy-pair mass at "
            f"{int(excess.sum())} pixels; retained because fractional CN "
            f"marginals do not define joint clone occupancy",
            RuntimeWarning)
    cis_rel = raw_cis_pairs
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
    visibility_band_cis_expected *= visibility_pair
    external_expected *= visibility_pair
    expected = cis_expected + external_expected
    usable = np.outer(valid_mask, valid_mask) & (expected > 0)
    oe = np.zeros_like(matrix, dtype=float)
    oe[usable] = matrix[usable] / expected[usable]
    return oe, GraphExpectedResult(
        expected, cis_expected, external_expected, cis_pairs,
        external_pairs, total_pairs, float(collision_floor), visibility,
        min_graph_distance, visibility_band_cis_expected)


def flow_residuals(segment_cn: np.ndarray, incident_junction_cn: np.ndarray,
                   source_cn: np.ndarray) -> np.ndarray:
    """Return segment-side flow residuals; source is explicit, not an SV edge."""
    target = np.repeat(np.asarray(segment_cn, float), 2)
    incident = np.asarray(incident_junction_cn, float).reshape(-1)
    source = np.asarray(source_cn, float).reshape(-1)
    if not (len(target) == len(incident) == len(source)):
        raise ValueError("flow arrays must describe both sides of every segment")
    return incident + source - target
