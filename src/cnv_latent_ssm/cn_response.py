"""Low-dimensional CN-state response on fixed graph physical exposure."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class CNResponseFit:
    knots: np.ndarray
    values: np.ndarray
    direction: str
    objective: float

    def evaluate(self, relative_cn):
        log_r = np.log(np.maximum(np.asarray(relative_cn, float), 1e-3))
        return np.exp(np.interp(log_r, self.knots, self.values))


def _fit_direction(observed, exposure, log_r_i, log_r_j, knots,
                   ridge, smooth, direction):
    def interpolate(parameters, values):
        return np.interp(values, knots, parameters)

    def objective(parameters):
        linear = (interpolate(parameters, log_r_i)
                  + interpolate(parameters, log_r_j))
        mu = np.maximum(exposure * np.exp(np.clip(linear, -10, 10)), 1e-12)
        loss = np.mean(mu - observed * np.log(mu))
        second = np.diff(parameters, n=2)
        penalty = ridge * np.mean(parameters ** 2)
        if len(second):
            penalty += smooth * np.mean(second ** 2)
        return loss + penalty

    constraints = []
    sign = 1 if direction == "increasing" else -1
    for i in range(len(knots) - 1):
        constraints.append({
            "type": "ineq",
            "fun": lambda x, i=i, sign=sign: sign * (x[i + 1] - x[i])})
    # Anchor one-copy effective yield to one. Interpolate because zero need not
    # coincide with a quantile-derived knot.
    constraints.append({
        "type": "eq", "fun": lambda x: np.interp(0.0, knots, x)})
    fit = minimize(objective, np.zeros(len(knots)), method="SLSQP",
                   constraints=constraints, bounds=[(-3, 3)] * len(knots),
                   options={"maxiter": 2000, "ftol": 1e-10})
    if not fit.success:
        raise RuntimeError(f"CN response fit failed: {fit.message}")
    return CNResponseFit(knots, fit.x, direction, float(fit.fun))


def fit_monotonic_cn_response(observed, exposure, cn_i, cn_j,
                              ridge=1.0, smooth=1.0, n_knots=5,
                              max_pixels=300_000, seed=180426):
    """Fit symmetric g(r_i,r_j)=exp[f(log r_i)+f(log r_j)]."""
    observed = np.asarray(observed, float)
    exposure = np.asarray(exposure, float)
    cn_i, cn_j = np.asarray(cn_i, float), np.asarray(cn_j, float)
    use = (np.isfinite(observed) & (observed >= 0) & np.isfinite(exposure)
           & (exposure > 0) & np.isfinite(cn_i) & np.isfinite(cn_j)
           & (cn_i > 0) & (cn_j > 0))
    ids = np.flatnonzero(use)
    if len(ids) > max_pixels:
        ids = np.random.default_rng(seed).choice(ids, max_pixels, replace=False)
    y, base = observed[ids], exposure[ids]
    li, lj = np.log(cn_i[ids]), np.log(cn_j[ids])
    pooled = np.r_[li, lj, 0.0]
    knots = np.unique(np.quantile(pooled, np.linspace(0, 1, n_knots)))
    if len(knots) < 3:
        raise ValueError("insufficient distinct CN states for response fit")
    fits = [_fit_direction(y, base, li, lj, knots, ridge, smooth, direction)
            for direction in ("increasing", "decreasing")]
    return min(fits, key=lambda x: x.objective)

