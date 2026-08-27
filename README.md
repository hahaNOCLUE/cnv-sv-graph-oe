# CNV-aware Latent Chromatin State Model

A modular probabilistic state-space modeling framework designed for robust, continuous A/B compartment calling in cancer Hi-C and Micro-C genomes harboring complex copy number variations (CNVs).

> **Development status:** the CNV/SV graph-aware expected-contact and
> derivative-walk scripts are active research code. Validate copy-flow
> conservation, junction dosage, and residual calibration on independent data
> before using their output for biological conclusions.

---

## 1. Mathematical Framework

The model disentangles genuine biological chromatin state dynamics from confounding CNV artifacts using a linear Gaussian State Space Model (SSM):

### State Equation (Latent Compartment Trajectory)
- **1D Continuous AR(1)**:
  $$x_i = F x_{i-1} + w_i, \quad w_i \sim \mathcal{N}(0, Q)$$
- **2D Local Linear Trend (Score + Velocity)**:
  $$\begin{bmatrix} s_i \\ v_i \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} s_{i-1} \\ v_{i-1} \end{bmatrix} + \mathbf{w}_i, \quad \mathbf{w}_i \sim \mathcal{N}(0, Q)$$

### Observation Equation
For top $K$ principal components $\mathbf{y}_i = (PC1_i, \dots, PCK_i)^T \in \mathbb{R}^K$:
$$\mathbf{y}_i = H x_i + G c_i + \mathbf{v}_i, \quad \mathbf{v}_i \sim \mathcal{N}(0, R)$$
where:
- $x_i$: continuous latent compartment activity.
- $H \in \mathbb{R}^{K \times 1}$: compartment loadings across observation features.
- $c_i = \log_2(CN_i / P)$: CNV deviation relative to sample baseline ploidy $P$.
- $G \in \mathbb{R}^{K \times 1}$: CNV effect vector capturing copy-number artifacts across features.
- $R = \operatorname{diag}(\sigma_1^2, \dots, \sigma_K^2)$: observation noise covariance.

---

## 2. Modular Architecture

```text
                     ┌── External CNV (WGS / WES / ASCAT / NeoLoopFinder)
Hi-C / Micro-C ──────┼── Internal CNV (1D Coverage -> GAM/Poly GC bias -> HMM segmentation)
                     │
                     ▼
                 CNV Module
                     │  c_i = log2(CN_i / P)
                     ▼
          Hi-C Feature Construction
          (O/E -> Correlation S -> PCA y_i)
                     │
                     ▼
               CNV-aware SSM
           (EM + Kalman Smoother)
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
 Continuous Score x_i     Discrete States
   & Posterior SD          A / B / U Calls
```

---

## 3. Output Files

- `*.compartment_score.bedGraph`: Continuous latent compartment score $\mathbb{E}[x_i \mid Y, CN]$.
- `*.compartment_state.bed`: Discrete states (A / B / U), posterior SD $\sqrt{\operatorname{Var}(x_i)}$, and confidence level.
- `*.cnv.bedGraph`: CNV deviation $c_i$.
- `*.parameters.tsv`: Estimated parameters ($F, Q, R, H, G$, log-likelihood, iterations).
- `*.qc.tsv`: Quantitative metrics (CNV correlation decoupling, gene density concordance, posterior uncertainty).
- `*.summary.png`: Multi-track visualization comparing input PCs, CNV, SSM trajectory with uncertainty ribbon, and loading decomposition.

---

## Copy-flow additive expected contacts

`cnv_latent_ssm.graph_expected` implements the rearranged-genome expected model

$$
E_{ij}=q_iq_j\left[\sum_p M_{ij,p}P_{0}(d_{ij,p})
       +B\,N^{\rm inter}_{ij}\right],
\qquad D_{ij}=\frac{CN_iCN_j}{P^2}.
$$

Its input is an oriented derivative-walk decomposition with explicit `strand`
and `walk_cn`. Consequently, junction dosage comes from the upstream JCN/flow
solution, deletion and inversion paths retain their physical orientation, and
multiple alleles are summed rather than collapsed to one shortest path. Source
or sink CN remains outside the derivative walks and therefore cannot create an
SV contact contribution. The constant collision floor $B$ is estimated from
graph-unconnected or genome-wide trans copy pairs; no CN exponent, compartment
state, or freely fitted long-range bin effect enters the expected model.

The older `compute_sv_distance_mixture_oe()` function is retained only for
backward compatibility and hop-cap sensitivity diagnostics. It is not the
recommended model for complex cancer genomes.
