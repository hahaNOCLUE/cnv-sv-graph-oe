"""State Space Model (SSM) Module: CNV-aware latent chromatin state estimation via EM + Kalman Smoother."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy import sparse
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


@dataclass
class SSMResults:
    """Results container for CNV-aware SSM model fitting."""
    chrom: str
    bins: pd.DataFrame
    # Latent state trajectory and uncertainty
    state_score: np.ndarray          # E[x_i | Y, CN]
    posterior_sd: np.ndarray         # sqrt(Var(x_i | Y, CN))
    # Discrete calling
    p_active: np.ndarray             # P(x_i > 0)
    discrete_state: np.ndarray       # 'A', 'B', or 'U'
    confidence: np.ndarray           # 'high', 'medium', 'low'
    # CNV deviation
    cnv_deviation: np.ndarray        # c_i = log2(CN_i / P)
    cnv_raw: np.ndarray              # Raw CN or relative ratio
    # Model parameters and state slope
    state_slope: Optional[np.ndarray] = None # E[v_i | Y, CN] for local linear trend
    local_artifact_score: Optional[np.ndarray] = None
    model_type: str = "ar1"
    F: Optional[np.ndarray] = None             # State transition matrix
    H: Optional[np.ndarray] = None             # Compartment loading (K x 1)
    G: Optional[np.ndarray] = None             # CNV effect vector (K x 1)
    Q: Optional[np.ndarray] = None             # Process noise covariance
    R: Optional[np.ndarray] = None             # Observation noise covariance (diagonal)
    log_likelihood: float = 0.0
    iterations: int = 0
    converged: bool = False
    loglik_history: Optional[list] = None
    explained_variance_ratio: Optional[np.ndarray] = None


class CNVAwareSSM:
    """CNV-aware Latent State Space Model for continuous compartment scoring.

    Models:
      State:       x_i = F x_{i-1} + w_i,   w_i ~ N(0, Q)
      Observation: y_i = H x_i + G c_i + v_i, v_i ~ N(0, R)
    """

    def __init__(
        self,
        model_type: str = "ar1",      # 'ar1' or 'local_linear_trend'
        n_pcs: int = 10,
        max_iter: int = 400,
        tol: float = 1e-5,
        min_var: float = 1e-5,
        loading_init: str = "pc1",
        freeze_cnv_effect: bool = False,
        sv_edge_strength: float = 0.0,
        contact_graph_strength: float = 0.0,
        verbose: bool = True,
    ):
        self.model_type = model_type
        self.n_pcs = n_pcs
        self.max_iter = max_iter
        self.tol = tol
        self.min_var = min_var
        if loading_init not in {"pc1", "uniform", "phasing"}:
            raise ValueError("loading_init must be 'pc1', 'uniform', or 'phasing'")
        self.loading_init = loading_init
        self.freeze_cnv_effect = freeze_cnv_effect
        if sv_edge_strength < 0:
            raise ValueError("sv_edge_strength must be non-negative")
        self.sv_edge_strength = sv_edge_strength
        self.contact_graph_strength = contact_graph_strength
        self.verbose = verbose

    def fit(
        self,
        y: np.ndarray,                # shape (T, K)
        c: np.ndarray,                # shape (T,)
        valid_mask: np.ndarray,       # shape (T,)
        prior_phasing_track: Optional[np.ndarray] = None, # shape (T,)
        breakpoints: Optional[np.ndarray] = None,         # shape (T,) boolean mask
        sv_edges: Optional[list] = None,
        contact_graph_edges: Optional[list] = None,
    ) -> Dict:
        """Fit the SSM using Expectation-Maximization (EM).

        Parameters
        ----------
        y : np.ndarray, shape (T, K)
            Observation matrix of top K principal components.
        c : np.ndarray, shape (T,)
            CNV deviation track c_i = log2(CN_i / P).
        valid_mask : np.ndarray, shape (T,)
            Boolean array of valid (unmasked) bins.
        prior_phasing_track : np.ndarray, optional
            Gene density or GC fraction used after fitting to orient the
            otherwise sign-indeterminate latent score. It also initializes H
            when ``loading_init='phasing'`` is explicitly selected.

        Returns
        -------
        results : dict containing estimated parameters, smoothed states, and log-likelihood.
        """
        T, K = y.shape
        is_llt = (self.model_type == "local_linear_trend")
        is_two_scale = (self.model_type == "two_scale")
        state_dim = 2 if (is_llt or is_two_scale) else 1

        # 1. Initialize parameters
        if is_two_scale:
            # Unit stationary variance at two distinct genomic scales.
            phi_global, phi_local = 0.995, 0.90
            F = np.diag([phi_global, phi_local])
            Q = np.diag([1.0 - phi_global ** 2, 1.0 - phi_local ** 2])
            mu0 = np.zeros(2)
            Sigma0 = np.eye(2)
        elif is_llt:
            F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=float)
            Q = np.diag([0.1, 0.01])
            mu0 = np.zeros(2)
            Sigma0 = np.eye(2) * 1.0
        else:
            F = np.array([[0.95]], dtype=float)
            Q = np.array([[0.1]], dtype=float)
            mu0 = np.zeros(1)
            Sigma0 = np.array([[1.0]], dtype=float)

        # Initialization is contact-only unless the explicitly requested
        # loading_init='phasing' mode supplies a biological selection prior.
        H = np.zeros((K, state_dim))
        if self.loading_init == "phasing":
            if prior_phasing_track is None:
                raise ValueError("loading_init='phasing' requires prior_phasing_track")
            init_valid = valid_mask & np.isfinite(prior_phasing_track)
            if np.sum(init_valid) < 3 or np.std(prior_phasing_track[init_valid]) < 1e-8:
                raise ValueError("phasing track has insufficient finite variation")
            for k in range(K):
                H[k, 0] = stats.pearsonr(
                    y[init_valid, k], prior_phasing_track[init_valid]
                )[0]
            h_norm = np.linalg.norm(H[:, 0])
            if not np.isfinite(h_norm) or h_norm < 1e-8:
                raise ValueError("phasing track has no usable correlation with observation PCs")
            H[:, 0] /= h_norm
        elif self.loading_init == "uniform":
            H[:, 0] = 1.0 / np.sqrt(K)
        else:
            H[0, 0] = 1.0
        if is_two_scale and K > 1:
            H[1, 1] = 0.5

        # Initial G from regression of y_k on c
        G = np.zeros((K, 1))
        c_valid = c[valid_mask]
        if np.std(c_valid) > 1e-4:
            for k in range(K):
                yk = y[valid_mask, k]
                slope, _, _, _, _ = stats.linregress(c_valid, yk)
                G[k, 0] = slope

        # Initial observation noise R (diagonal)
        R_diag = np.ones(K, dtype=float)
        for k in range(K):
            yk = y[valid_mask, k]
            R_diag[k] = max(np.var(yk) * 0.5, self.min_var)
        R = np.diag(R_diag)

        loglik_history = []
        converged = False

        # EM loop
        for it in range(self.max_iter):
            # ----------------------------------------------------
            # E-step: Forward Kalman Filter + Backward RTS Smoother
            # ----------------------------------------------------
            smoothed_means, smoothed_covs, lag1_covs, loglik = self._kalman_smoother(
                y=y,
                c=c,
                valid_mask=valid_mask,
                F=F,
                H=H,
                G=G,
                Q=Q,
                R=R,
                mu0=mu0,
                Sigma0=Sigma0,
                breakpoints=breakpoints,
            )
            if sv_edges and self.sv_edge_strength > 0:
                smoothed_means = self._apply_sv_graph_transitions(
                    smoothed_means, smoothed_covs, F, Q, sv_edges,
                )
            if contact_graph_edges and self.contact_graph_strength > 0:
                smoothed_means = self._apply_contact_graph(
                    smoothed_means, smoothed_covs, Q, contact_graph_edges,
                )
            loglik_history.append(loglik)

            if it > 0:
                d_ll = loglik - loglik_history[-2]
                rel_change = abs(d_ll) / (abs(loglik_history[-2]) + 1.0)
                if abs(d_ll) < self.tol or rel_change < self.tol or (d_ll < 0 and abs(d_ll) < self.tol * 10):
                    converged = True
                    if self.verbose:
                        logger.info(f"EM converged at iteration {it+1} (logLik = {loglik:.4f}, delta = {d_ll:.6f})")
                    break

            # ----------------------------------------------------
            # Identifiability Normalization: Scale Var(x) = 1
            # ----------------------------------------------------
            x_smoothed_s = smoothed_means[:, 0]
            valid_x = x_smoothed_s[valid_mask]
            std_x = np.std(valid_x)
            if std_x > 1e-6 and not is_two_scale:
                scale = std_x
                smoothed_means[:, 0] /= scale
                smoothed_covs[:, 0, 0] /= (scale ** 2)
                lag1_covs[:, 0, 0] /= (scale ** 2)
                H[:, 0] *= scale
                if is_llt:
                    smoothed_means[:, 1] /= scale
                    smoothed_covs[:, 1, 1] /= (scale ** 2)
                    smoothed_covs[:, 0, 1] /= (scale ** 2)
                    smoothed_covs[:, 1, 0] /= (scale ** 2)
                    lag1_covs[:, 1, 1] /= (scale ** 2)
                    lag1_covs[:, 0, 1] /= (scale ** 2)
                    lag1_covs[:, 1, 0] /= (scale ** 2)

            # ----------------------------------------------------
            # M-step: Parameter Updates
            # ----------------------------------------------------
            # 1. Update State Transition F & Process Noise Q
            trans_mask = np.ones(T - 1, dtype=bool)
            if breakpoints is not None:
                trans_mask = trans_mask & (~breakpoints[1:])

            if is_two_scale:
                # F and Q stay fixed so the global/local time scales cannot
                # rotate into one another during EM.
                pass
            elif not is_llt:
                # 1D AR(1): F is scalar
                sum_cross = np.sum((lag1_covs[1:, 0, 0] + smoothed_means[1:, 0] * smoothed_means[:-1, 0])[trans_mask])
                sum_prev_sq = np.sum((smoothed_covs[:-1, 0, 0] + (smoothed_means[:-1, 0] ** 2))[trans_mask])
                if sum_prev_sq > 1e-12:
                    F[0, 0] = np.clip(sum_cross / sum_prev_sq, -0.999, 0.999)

                # Q for AR(1)
                sum_curr_sq = np.sum((smoothed_covs[1:, 0, 0] + (smoothed_means[1:, 0] ** 2))[trans_mask])
                n_trans = max(np.sum(trans_mask), 1)
                q_val = (sum_curr_sq - 2 * F[0, 0] * sum_cross + (F[0, 0] ** 2) * sum_prev_sq) / n_trans
                Q[0, 0] = max(q_val, self.min_var)
            else:
                # Local Linear Trend: F is fixed [[1,1],[0,1]], update Q = diag(q_s, q_v)
                res_cov = np.zeros((2, 2))
                for t in range(1, T):
                    if breakpoints is not None and breakpoints[t]:
                        continue
                    E_curr_sq = smoothed_covs[t] + np.outer(smoothed_means[t], smoothed_means[t])
                    E_prev_sq = smoothed_covs[t-1] + np.outer(smoothed_means[t-1], smoothed_means[t-1])
                    E_cross = lag1_covs[t] + np.outer(smoothed_means[t], smoothed_means[t-1])

                    res_t = E_curr_sq - F @ E_cross.T - E_cross @ F.T + F @ E_prev_sq @ F.T
                    res_cov += res_t

                n_trans = max(np.sum(trans_mask), 1)
                res_cov /= n_trans
                Q = np.diag(np.maximum(np.diag(res_cov), self.min_var))

            # 2. Update Observation Loadings H and CNV effect G
            # Observation model: y_i = H x_i + G c_i + v_i
            # Stack regressors: u_i = [s_i, (v_i if LLT), c_i]^T
            n_reg = state_dim + 1
            sum_uu = np.zeros((n_reg, n_reg))
            sum_yu = np.zeros((K, n_reg))
            n_valid = 0

            for t in range(T):
                if not valid_mask[t]:
                    continue
                n_valid += 1
                s_t = smoothed_means[t]
                P_t = smoothed_covs[t]
                c_t = c[t]

                # E[u_t]
                E_u = np.zeros(n_reg)
                E_u[:state_dim] = s_t
                E_u[state_dim] = c_t

                # E[u_t u_t^T]
                E_uu = np.zeros((n_reg, n_reg))
                E_uu[:state_dim, :state_dim] = P_t + np.outer(s_t, s_t)
                E_uu[:state_dim, state_dim] = s_t * c_t
                E_uu[state_dim, :state_dim] = s_t * c_t
                E_uu[state_dim, state_dim] = c_t ** 2

                sum_uu += E_uu
                sum_yu += np.outer(y[t], E_u)

            if n_valid > 10:
                if self.freeze_cnv_effect:
                    # Two-stage fit: G was estimated before EM. Regress the
                    # CN-residualized observations only on the latent state,
                    # so CN signal cannot move back from G into Hx.
                    sum_xx = sum_uu[:state_dim, :state_dim]
                    sum_yx = sum_yu[:, :state_dim]
                    sum_cx = sum_uu[state_dim, :state_dim]
                    residual_yx = sum_yx - G @ sum_cx[None, :]
                    H = residual_yx @ np.linalg.inv(
                        sum_xx + np.eye(state_dim) * 1e-5
                    )
                    W = np.column_stack([H, G])
                else:
                    # Joint update, regularized against numerical collinearity.
                    reg_inv = np.linalg.inv(sum_uu + np.eye(n_reg) * 1e-5)
                    W = sum_yu @ reg_inv   # shape (K, n_reg)
                    H = W[:, :state_dim]
                    G = W[:, state_dim:state_dim+1]

                if is_two_scale:
                    # Separate the fast local observation direction from the
                    # global direction in the current noise metric.
                    r_inv = 1.0 / np.maximum(np.diag(R), self.min_var)
                    hg = H[:, 0]
                    hl = H[:, 1]
                    denom = np.sum(r_inv * hg * hg)
                    if denom > 1e-12:
                        H[:, 1] = hl - hg * (np.sum(r_inv * hg * hl) / denom)
                    W[:, :state_dim] = H

                # 3. Update Observation Noise Covariance R (diagonal)
                R_diag_new = np.zeros(K)
                for t in range(T):
                    if not valid_mask[t]:
                        continue
                    y_t = y[t]
                    s_t = smoothed_means[t]
                    P_t = smoothed_covs[t]
                    c_t = c[t]

                    E_u = np.zeros(n_reg)
                    E_u[:state_dim] = s_t
                    E_u[state_dim] = c_t

                    E_uu = np.zeros((n_reg, n_reg))
                    E_uu[:state_dim, :state_dim] = P_t + np.outer(s_t, s_t)
                    E_uu[:state_dim, state_dim] = s_t * c_t
                    E_uu[state_dim, :state_dim] = s_t * c_t
                    E_uu[state_dim, state_dim] = c_t ** 2

                    # diag( y y^T - 2 y E[u]^T W^T + W E[uu] W^T )
                    resid_sq = (y_t ** 2) - 2 * y_t * (W @ E_u) + np.diag(W @ E_uu @ W.T)
                    R_diag_new += resid_sq

                R_diag_new /= max(n_valid, 1)
                R = np.diag(np.maximum(R_diag_new, self.min_var))

        # Final smoothing pass with converged parameters
        smoothed_means, smoothed_covs, _, final_ll = self._kalman_smoother(
            y=y,
            c=c,
            valid_mask=valid_mask,
            F=F,
            H=H,
            G=G,
            Q=Q,
            R=R,
            mu0=mu0,
            Sigma0=Sigma0,
            breakpoints=breakpoints,
        )
        if sv_edges and self.sv_edge_strength > 0:
            smoothed_means = self._apply_sv_graph_transitions(
                smoothed_means, smoothed_covs, F, Q, sv_edges,
            )
        if contact_graph_edges and self.contact_graph_strength > 0:
            smoothed_means = self._apply_contact_graph(
                smoothed_means, smoothed_covs, Q, contact_graph_edges,
            )

        state_score = smoothed_means[:, 0].copy()
        posterior_sd = np.sqrt(np.maximum(smoothed_covs[:, 0, 0], 1e-6))
        state_slope = smoothed_means[:, 1].copy() if is_llt else None
        local_artifact_score = smoothed_means[:, 1].copy() if is_two_scale else None

        # ----------------------------------------------------
        # Phasing / Sign Orientation Alignment
        # ----------------------------------------------------
        if prior_phasing_track is not None:
            val_both = valid_mask & np.isfinite(state_score) & np.isfinite(prior_phasing_track)
            if np.sum(val_both) > 10:
                corr_sign, _ = stats.pearsonr(state_score[val_both], prior_phasing_track[val_both])
                if corr_sign < 0:
                    if self.verbose:
                        logger.info(f"Flipping latent compartment sign (corr with prior track = {corr_sign:.3f})")
                    state_score = -state_score
                    H = -H
                    if state_slope is not None:
                        state_slope = -state_slope

        # ----------------------------------------------------
        # Discrete State & Confidence Calling
        # ----------------------------------------------------
        z_scores = state_score / np.maximum(posterior_sd, 1e-6)
        p_active = stats.norm.cdf(z_scores)

        discrete_state = np.full(T, "U", dtype=object)
        confidence = np.full(T, "low", dtype=object)

        # High confidence calling (p > 0.95 or p < 0.05)
        high_a = p_active >= 0.95
        high_b = p_active <= 0.05
        discrete_state[high_a] = "A"
        confidence[high_a] = "high"
        discrete_state[high_b] = "B"
        confidence[high_b] = "high"

        # Medium confidence calling
        med_a = (p_active >= 0.70) & (~high_a)
        med_b = (p_active <= 0.30) & (~high_b)
        discrete_state[med_a] = "A"
        confidence[med_a] = "medium"
        discrete_state[med_b] = "B"
        confidence[med_b] = "medium"

        # Invalid mask sets NaNs
        state_score[~valid_mask] = np.nan
        if local_artifact_score is not None:
            local_artifact_score[~valid_mask] = np.nan
        posterior_sd[~valid_mask] = np.nan
        p_active[~valid_mask] = np.nan
        discrete_state[~valid_mask] = "NA"
        confidence[~valid_mask] = "NA"

        return {
            "state_score": state_score,
            "posterior_sd": posterior_sd,
            "state_slope": state_slope,
            "local_artifact_score": local_artifact_score,
            "p_active": p_active,
            "discrete_state": discrete_state,
            "confidence": confidence,
            "F": F,
            "H": H,
            "G": G,
            "Q": Q,
            "R": R,
            "log_likelihood": final_ll,
            "iterations": len(loglik_history),
            "converged": converged,
            "loglik_history": loglik_history,
        }

    def _apply_sv_graph_transitions(
        self,
        means: np.ndarray,
        covs: np.ndarray,
        F: np.ndarray,
        Q: np.ndarray,
        sv_edges: list,
    ) -> np.ndarray:
        """Gaussian graph update for paired SV endpoint transitions.

        The Kalman posterior supplies a local Gaussian term at every bin. Each
        EagleC2 edge adds ``x_j - F x_i ~ N(0, Q / strength)``. Solving the
        resulting sparse precision system couples the paired endpoints while
        retaining the chromosome-chain posterior as the baseline.
        """
        T, state_dim = means.shape
        updated = means.copy()
        for d in range(state_dim):
            marginal_var = np.maximum(covs[:, d, d], self.min_var)
            base_precision = 1.0 / marginal_var
            A = sparse.diags(base_precision, format="lil")
            b = base_precision * means[:, d]
            f_edge = float(F[d, d])
            edge_precision = self.sv_edge_strength / max(float(Q[d, d]), self.min_var)
            for edge in sv_edges:
                i, j = edge[:2]
                edge_weight = float(edge[2]) if len(edge) > 2 else 1.0
                if i == j or not (0 <= i < T and 0 <= j < T):
                    continue
                weighted_precision = edge_precision * edge_weight
                A[i, i] += weighted_precision * f_edge * f_edge
                A[j, j] += weighted_precision
                cross = weighted_precision * f_edge
                A[i, j] -= cross
                A[j, i] -= cross
            updated[:, d] = spsolve(A.tocsr(), b)
        return updated

    def _apply_contact_graph(
        self, means: np.ndarray, covs: np.ndarray, Q: np.ndarray, edges: list,
    ) -> np.ndarray:
        """Constrain only the global state using distal contact similarity edges."""
        updated = means.copy()
        variance = np.maximum(covs[:, 0, 0], self.min_var)
        base_precision = 1.0 / variance
        A = sparse.diags(base_precision, format="lil")
        b = base_precision * means[:, 0]
        graph_precision = self.contact_graph_strength / max(float(Q[0, 0]), self.min_var)
        for edge in edges:
            i, j = edge[:2]
            weight = float(edge[2]) if len(edge) > 2 else 1.0
            relation = float(edge[3]) if len(edge) > 3 else 1.0
            precision = graph_precision * weight
            A[i, i] += precision
            A[j, j] += precision
            A[i, j] -= precision * relation
            A[j, i] -= precision * relation
        updated[:, 0] = spsolve(A.tocsr(), b)
        return updated

    def _kalman_smoother(
        self,
        y: np.ndarray,
        c: np.ndarray,
        valid_mask: np.ndarray,
        F: np.ndarray,
        H: np.ndarray,
        G: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
        breakpoints: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Kalman filter and RTS smoother with full or local-only breakpoint resets."""
        T, K = y.shape
        state_dim = F.shape[0]
        is_two_scale = self.model_type == "two_scale"

        # Allocate forward pass containers
        pred_means = np.zeros((T, state_dim))
        pred_covs = np.zeros((T, state_dim, state_dim))
        filt_means = np.zeros((T, state_dim))
        filt_covs = np.zeros((T, state_dim, state_dim))
        F_steps = np.zeros((T, state_dim, state_dim))

        loglik = 0.0
        R_inv = np.linalg.inv(R)
        R_det = np.linalg.det(R)

        # ----------------- Forward Pass -----------------
        for t in range(T):
            if t == 0:
                x_pred = mu0.copy()
                P_pred = Sigma0.copy()
            elif is_two_scale and breakpoints is not None and breakpoints[t]:
                F_t = F.copy()
                F_t[1, :] = 0.0
                Q_t = Q.copy()
                Q_t[1, 1] = Sigma0[1, 1]
                F_steps[t] = F_t
                x_pred = F_t @ filt_means[t - 1]
                P_pred = F_t @ filt_covs[t - 1] @ F_t.T + Q_t
            elif breakpoints is not None and breakpoints[t]:
                x_pred = mu0.copy()
                P_pred = Sigma0.copy()
            else:
                F_steps[t] = F
                x_pred = F @ filt_means[t - 1]
                P_pred = F @ filt_covs[t - 1] @ F.T + Q

            # Symmetrize
            P_pred = 0.5 * (P_pred + P_pred.T)
            pred_means[t] = x_pred
            pred_covs[t] = P_pred

            if not valid_mask[t]:
                # Missing observation: no update
                filt_means[t] = x_pred
                filt_covs[t] = P_pred
            else:
                # Observed bin
                y_t = y[t]
                c_t = c[t]
                y_pred = H @ x_pred + (G @ np.array([c_t])).flatten()
                v_t = y_t - y_pred  # innovation

                # Innovation covariance S = H P_pred H^T + R
                S_t = H @ P_pred @ H.T + R
                S_t = 0.5 * (S_t + S_t.T)
                S_t_inv = np.linalg.inv(S_t)

                # Kalman gain
                K_gain = P_pred @ H.T @ S_t_inv  # shape (state_dim, K)

                filt_means[t] = x_pred + K_gain @ v_t
                filt_covs[t] = (np.eye(state_dim) - K_gain @ H) @ P_pred
                filt_covs[t] = 0.5 * (filt_covs[t] + filt_covs[t].T)

                # Gaussian log-likelihood
                sign, logdet = np.linalg.slogdet(S_t)
                quad = v_t @ S_t_inv @ v_t
                loglik += -0.5 * (K * np.log(2 * np.pi) + logdet + quad)

        # ----------------- Backward RTS Smoother -----------------
        smoothed_means = np.zeros((T, state_dim))
        smoothed_covs = np.zeros((T, state_dim, state_dim))
        lag1_covs = np.zeros((T, state_dim, state_dim))
        J = np.zeros((T - 1, state_dim, state_dim))

        smoothed_means[-1] = filt_means[-1]
        smoothed_covs[-1] = filt_covs[-1]

        for t in range(T - 2, -1, -1):
            if breakpoints is not None and breakpoints[t + 1] and not is_two_scale:
                # Breakpoint boundary: decouple backward state smoothing
                smoothed_means[t] = filt_means[t]
                smoothed_covs[t] = filt_covs[t]
                J[t] = np.zeros((state_dim, state_dim))
            else:
                P_pred_inv = np.linalg.inv(pred_covs[t + 1] + np.eye(state_dim) * 1e-8)
                F_next = F_steps[t + 1]
                J_t = filt_covs[t] @ F_next.T @ P_pred_inv
                J[t] = J_t

                smoothed_means[t] = filt_means[t] + J_t @ (smoothed_means[t + 1] - pred_means[t + 1])
                smoothed_covs[t] = filt_covs[t] + J_t @ (smoothed_covs[t + 1] - pred_covs[t + 1]) @ J_t.T
                smoothed_covs[t] = 0.5 * (smoothed_covs[t] + smoothed_covs[t].T)

        # Lag-1 cross-covariance Cov(x_t, x_{t-1} | Y)
        if valid_mask[-1]:
            S_last_inv = np.linalg.inv(H @ pred_covs[-1] @ H.T + R)
            K_last = pred_covs[-1] @ H.T @ S_last_inv
            lag1_covs[-1] = (np.eye(state_dim) - K_last @ H) @ F_steps[-1] @ filt_covs[-2]
        else:
            lag1_covs[-1] = F_steps[-1] @ filt_covs[-2]

        for t in range(T - 2, 0, -1):
            if breakpoints is not None and breakpoints[t] and not is_two_scale:
                lag1_covs[t] = np.zeros((state_dim, state_dim))
            else:
                lag1_covs[t] = filt_covs[t] @ J[t - 1].T + J[t] @ (lag1_covs[t + 1] - F_steps[t + 1] @ filt_covs[t]) @ J[t - 1].T

        return smoothed_means, smoothed_covs, lag1_covs, loglik
