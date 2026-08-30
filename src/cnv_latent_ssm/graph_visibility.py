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
    visibility_band_cis_expected: np.ndarray,
    valid_bins: np.ndarray,
    excluded_bins: np.ndarray | None = None,
    excluded_pairs: np.ndarray | None = None,
    minimum_band_fraction: float = 0.9,
    enrichment_quantile: float = 0.995,
) -> np.ndarray:
    """Select pixels dominated by 0.5--5 Mb same-molecule exposure."""
    valid_bins = np.asarray(valid_bins, bool)
    if excluded_bins is None:
        excluded_bins = np.zeros(len(valid_bins), bool)
    if excluded_pairs is None:
        excluded_pairs = np.zeros_like(observed, bool)
    eligible_bins = valid_bins & ~np.asarray(excluded_bins, bool)
    band_fraction = np.divide(
        visibility_band_cis_expected, graph_expected,
        out=np.zeros_like(graph_expected, float), where=graph_expected > 0)
    mask = (
        np.outer(eligible_bins, eligible_bins)
        & ~np.asarray(excluded_pairs, bool)
        & np.isfinite(observed)
        & (observed >= 0)
        & np.isfinite(graph_expected)
        & (graph_expected > 0)
        & np.isfinite(band_fraction)
        & (band_fraction >= minimum_band_fraction)
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


def split_visibility_mask(mask: np.ndarray, holdout_fraction: float = 0.2):
    """Deterministically split symmetric pair pixels into train and holdout."""
    if not 0 <= holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie in [0, 1)")
    mask = np.asarray(mask, bool)
    rows, cols = np.indices(mask.shape)
    # A symmetric integer hash avoids run-to-run random split variation.
    left, right = np.minimum(rows, cols), np.maximum(rows, cols)
    hashed = (left * 73856093 + right * 19349663) % 10_000
    holdout = mask & (hashed < int(round(10_000 * holdout_fraction)))
    train = mask & ~holdout
    return train, holdout


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
        previous_q = q.copy()
        previous_scale = scale
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
        parameter_error = max(
            float(np.max(np.abs(np.log(q[usable] / previous_q[usable])))),
            abs(float(np.log(scale / previous_scale))),
        )
        # With bounded q, row residuals can remain nonzero at the constrained
        # optimum.  Treat a numerically stationary parameter vector as
        # converged rather than exhausting max_iterations indefinitely.
        if max_error < tolerance or parameter_error < 1e-8:
            converged = True
            break

    return GraphVisibilityResult(
        q, scale, fit_mask, supported, converged, iteration, max_error,
        np.asarray(history, float),
    )
