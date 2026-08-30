"""CN-targeted iterative balancing for raw contact matrices."""
from dataclasses import dataclass

import numpy as np


@dataclass
class CNAwareBalanceResult:
    bias: np.ndarray
    target_scale: float
    iterations: int
    converged: bool
    max_abs_log_ratio: float
    history: list


def fit_cn_aware_balance(observed, copy_number, valid_bins, pair_mask=None,
                         damping=.5, bias_min=.5, bias_max=2.,
                         tolerance=.01, max_iterations=200):
    """Fit ``b`` so rows of ``observed / outer(b, b)`` scale with CN.

    The fit uses raw contacts only. Copy number defines the target marginal;
    it is not fitted as a pairwise contact response.
    """
    observed = np.asarray(observed, float)
    cn = np.asarray(copy_number, float)
    valid = np.asarray(valid_bins, bool)
    if observed.shape != (len(cn), len(cn)):
        raise ValueError("observed and copy-number dimensions differ")
    if pair_mask is None:
        pair_mask = np.outer(valid, valid)
        np.fill_diagonal(pair_mask, False)
    else:
        pair_mask = np.asarray(pair_mask, bool) & np.outer(valid, valid)
    observed_mass = np.where(pair_mask & np.isfinite(observed), observed, 0.).sum(axis=1)
    supported = valid & (cn > 0) & pair_mask.any(axis=1) & (observed_mass > 0)
    if supported.sum() < 2:
        raise ValueError("fewer than two bins have CN-aware balancing support")

    bias = np.ones(len(cn), float)
    history = []
    converged = False
    max_error = np.inf
    target_scale = 1.
    for iteration in range(1, max_iterations + 1):
        corrected = np.divide(
            observed, np.outer(bias, bias), out=np.zeros_like(observed),
            where=pair_mask & np.isfinite(observed))
        marginal = np.where(pair_mask, corrected, 0.).sum(axis=1)
        # Match total target mass to the current corrected matrix.  A geometric
        # center is inappropriate here because it changes the requested total
        # marginal and can prevent the symmetric balancing problem converging.
        target_scale = float(marginal[supported].sum() / cn[supported].sum())
        ratio = np.ones(len(cn), float)
        ratio[supported] = marginal[supported] / (target_scale * cn[supported])
        max_error = float(np.max(np.abs(np.log(np.maximum(ratio[supported], 1e-12)))))
        history.append((iteration, target_scale, max_error))
        if max_error < tolerance:
            converged = True
            break
        bias[supported] *= np.maximum(ratio[supported], 1e-12) ** (damping / 2.)
        bias[supported] = np.clip(bias[supported], bias_min, bias_max)
        center = float(np.median(np.log(bias[supported])))
        bias[supported] = np.clip(bias[supported] / np.exp(center), bias_min, bias_max)

    return CNAwareBalanceResult(
        bias, target_scale, iteration, converged, max_error, history)
