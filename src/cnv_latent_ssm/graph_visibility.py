"""Graph-aware iterative estimation of bin-specific technical visibility."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GraphVisibilityResult:
    visibility: np.ndarray
    scale: float
    fit_mask: np.ndarray
    supported_bins: np.ndarray
    converged: bool
    iterations: int
    max_abs_log_ratio: float
    history: np.ndarray


def build_graph_visibility_mask(
    observed: np.ndarray,
    graph_expected: np.ndarray,
    cis_expected: np.ndarray,
    graph_distance_bins: np.ndarray,
    valid_bins: np.ndarray,
    resolution: int,
    excluded_bins: np.ndarray | None = None,
    min_distance_bp: int = 500_000,
    max_distance_bp: int = 5_000_000,
    enrichment_quantile: float = 0.995,
) -> np.ndarray:
    """Select reliable graph-cis pixels for visibility estimation."""
    valid_bins = np.asarray(valid_bins, bool)
    if excluded_bins is None:
        excluded_bins = np.zeros(len(valid_bins), bool)
    eligible_bins = valid_bins & ~np.asarray(excluded_bins, bool)
    distance_bp = np.asarray(graph_distance_bins, float) * resolution
    mask = (
        np.outer(eligible_bins, eligible_bins)
        & np.isfinite(observed)
        & (observed >= 0)
        & np.isfinite(graph_expected)
        & (graph_expected > 0)
        & np.isfinite(distance_bp)
        & (distance_bp >= min_distance_bp)
        & (distance_bp <= max_distance_bp)
        & (cis_expected > 0)
    )
    np.fill_diagonal(mask, False)
    if not np.any(mask):
        raise ValueError("no reliable graph-cis pixels for visibility fitting")

    if not 0 < enrichment_quantile <= 1:
        raise ValueError("enrichment_quantile must lie in (0, 1]")
    if enrichment_quantile < 1:
        initial_ratio = np.divide(
            observed, graph_expected, out=np.zeros_like(observed, float),
            where=graph_expected > 0,
        )
        positive = initial_ratio[mask & (initial_ratio > 0)]
        if len(positive):
            upper = float(np.quantile(positive, enrichment_quantile))
            mask &= initial_ratio <= upper
    # Make the mask exactly symmetric even if a sparse input was asymmetric.
    return mask & mask.T


def fit_graph_visibility(
    observed: np.ndarray,
    graph_expected: np.ndarray,
    fit_mask: np.ndarray,
    valid_bins: np.ndarray,
    damping: float = 0.5,
    q_min: float = 0.5,
    q_max: float = 2.0,
    tolerance: float = 0.01,
    max_iterations: int = 100,
) -> GraphVisibilityResult:
    """Fit ``mu=s*q_i*q_j*E_graph`` by damped graph-aware balancing."""
    observed = np.asarray(observed, float)
    graph_expected = np.asarray(graph_expected, float)
    fit_mask = np.asarray(fit_mask, bool)
    valid_bins = np.asarray(valid_bins, bool)
    if observed.shape != graph_expected.shape or observed.shape != fit_mask.shape:
        raise ValueError("observed, graph_expected and fit_mask must match")
    if observed.shape != (len(valid_bins), len(valid_bins)):
        raise ValueError("valid_bins does not match matrix dimensions")
    if not 0 < damping <= 1:
        raise ValueError("damping must lie in (0, 1]")
    if not 0 < q_min <= 1 <= q_max:
        raise ValueError("visibility bounds must satisfy 0 < q_min <= 1 <= q_max")

    observed_row0 = np.where(fit_mask, observed, 0).sum(axis=1)
    supported = valid_bins & fit_mask.any(axis=1) & (observed_row0 > 0)
    if supported.sum() < 2:
        raise ValueError("fewer than two bins have visibility-fit support")
    fit_mask = fit_mask & np.outer(supported, supported)
    q = np.ones(len(valid_bins), float)
    exposure0 = float(graph_expected[fit_mask].sum())
    if exposure0 <= 0:
        raise ValueError("visibility fit has zero graph exposure")
    scale = float(observed[fit_mask].sum() / exposure0)
    scale = max(scale, 1e-12)
    history = []
    converged = False

    for iteration in range(1, max_iterations + 1):
        mu = scale * np.outer(q, q) * graph_expected
        observed_row = np.where(fit_mask, observed, 0).sum(axis=1)
        expected_row = np.where(fit_mask, mu, 0).sum(axis=1)
        ratio = np.ones(len(q), float)
        usable = supported & (expected_row > 0)
        ratio[usable] = observed_row[usable] / expected_row[usable]
        ratio[usable] = np.maximum(ratio[usable], 1e-12)
        # A bounded q cannot remove a residual that points beyond its active
        # bound. Treat those bins as constrained solutions, while retaining
        # their residuals in downstream row-CV diagnostics.
        adjustable = usable.copy()
        adjustable &= ~((q <= q_min * (1 + 1e-8)) & (ratio < 1))
        adjustable &= ~((q >= q_max * (1 - 1e-8)) & (ratio > 1))
        max_error = (float(np.max(np.abs(np.log(ratio[adjustable]))))
                     if adjustable.any() else 0.0)

        q[usable] *= np.power(ratio[usable], damping / 2.0)
        q[usable] = np.clip(q[usable], q_min, q_max)
        # Separate visibility from library scale by fixing median(log q)=0.
        q[usable] /= np.exp(np.median(np.log(q[usable])))
        q[usable] = np.clip(q[usable], q_min, q_max)
        q[~supported] = 1.0

        denominator = (np.outer(q, q) * graph_expected)[fit_mask].sum()
        scale = float(observed[fit_mask].sum() / max(denominator, 1e-12))
        history.append((iteration, scale, max_error))
        if max_error < tolerance:
            converged = True
            break

    return GraphVisibilityResult(
        q, scale, fit_mask, supported, converged, iteration, max_error,
        np.asarray(history, float),
    )
