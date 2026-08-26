"""Unit tests for CNV-aware Latent SSM."""

import numpy as np
import pytest
from scipy import stats
from cnv_latent_ssm.ssm import CNVAwareSSM
from cnv_latent_ssm.caller import (
    build_contact_knn_graph,
    identify_transition_breakpoints,
    load_sv_endpoint_breakpoints,
    weight_sv_edges_by_cnv,
)
from cnv_latent_ssm.cnv import get_cnv_track, load_external_cnv
from cnv_latent_ssm.features import compute_sv_distance_mixture_oe, identify_valid_bins
from cnv_latent_ssm.io import calder_style_rank_score


def test_ssm_ar1_synthetic():
    np.random.seed(42)
    T = 200
    K = 5
    
    # Generate true latent state
    F_true = 0.92
    Q_true = 0.15
    x_true = np.zeros(T)
    x_true[0] = np.random.normal(0, 1)
    for t in range(1, T):
        x_true[t] = F_true * x_true[t - 1] + np.random.normal(0, np.sqrt(Q_true))
    
    # Scale true x to unit variance
    x_true = (x_true - np.mean(x_true)) / np.std(x_true)

    # Generate synthetic CNV (e.g. amplification in middle)
    c = np.zeros(T)
    c[50:120] = 1.5  # log2(CN=4/2) = 1.0, 1.5
    
    # True loadings and CNV effects
    H_true = np.array([[0.8], [0.5], [-0.3], [0.1], [0.0]])
    G_true = np.array([[1.2], [0.1], [0.0], [0.8], [0.2]])
    R_true = np.diag([0.05, 0.05, 0.05, 0.05, 0.05])
    
    # Generate observations
    y = np.zeros((T, K))
    for t in range(T):
        y[t] = (H_true @ np.array([x_true[t]]) + G_true @ np.array([c[t]])).flatten() + np.random.multivariate_normal(np.zeros(K), R_true)
        
    valid_mask = np.ones(T, dtype=bool)
    valid_mask[10:15] = False  # Mask a small gap

    ssm = CNVAwareSSM(model_type="ar1", n_pcs=K, max_iter=100, tol=1e-5, verbose=False)
    results = ssm.fit(y=y, c=c, valid_mask=valid_mask, prior_phasing_track=x_true)
    
    x_est = results["state_score"]
    val = valid_mask & np.isfinite(x_est)
    
    # Correlation between true x and estimated x should be very high (>0.85)
    r = np.corrcoef(x_true[val], x_est[val])[0, 1]
    assert r > 0.85, f"Expected correlation > 0.85, got {r:.3f}"
    assert results["converged"], "EM did not converge"
    assert results["iterations"] < 100


def test_ssm_local_linear_trend_synthetic():
    np.random.seed(42)
    T = 150
    K = 4
    
    # Local linear trend state simulation
    x_true = np.zeros(T)
    v_true = np.zeros(T)
    v_true[0] = 0.02
    x_true[0] = 1.0
    for t in range(1, T):
        v_true[t] = v_true[t - 1] + np.random.normal(0, 0.01)
        x_true[t] = x_true[t - 1] + v_true[t - 1] + np.random.normal(0, 0.05)
    
    x_true = (x_true - np.mean(x_true)) / np.std(x_true)
    c = np.zeros(T)
    c[30:70] = 1.0
    
    H_true = np.array([[1.0], [0.4], [-0.2], [0.0]])
    G_true = np.array([[0.8], [0.0], [0.5], [0.1]])
    
    y = np.zeros((T, K))
    for t in range(T):
        y[t] = (H_true @ np.array([x_true[t]]) + G_true @ np.array([c[t]])).flatten() + np.random.normal(0, 0.1, size=K)
        
    valid_mask = np.ones(T, dtype=bool)

    ssm = CNVAwareSSM(model_type="local_linear_trend", n_pcs=K, max_iter=100, tol=1e-5, verbose=False)
    results = ssm.fit(y=y, c=c, valid_mask=valid_mask, prior_phasing_track=x_true)
    
    x_est = results["state_score"]
    val = valid_mask & np.isfinite(x_est)
    
    r = np.corrcoef(x_true[val], x_est[val])[0, 1]
    assert r > 0.80, f"Expected correlation > 0.80, got {r:.3f}"
    assert results["state_slope"] is not None


def test_ssm_decoupled_breakpoints():
    np.random.seed(123)
    T = 100
    K = 3
    x_true = np.random.normal(0, 1, T)
    c = np.zeros(T)
    c[50:] = 1.0
    
    y = np.random.normal(0, 1, (T, K))
    valid_mask = np.ones(T, dtype=bool)
    breakpoints = np.zeros(T, dtype=bool)
    breakpoints[0] = True
    breakpoints[50] = True
    
    ssm = CNVAwareSSM(model_type="ar1", n_pcs=K, max_iter=20, tol=1e-4, verbose=False)
    results = ssm.fit(y=y, c=c, valid_mask=valid_mask, prior_phasing_track=x_true, breakpoints=breakpoints)
    assert results["state_score"] is not None
    assert len(results["state_score"]) == T


def test_multi_sv_distance_uses_nearby_compound_edges():
    matrix = np.ones((12, 12), dtype=float)
    valid = np.ones(12, dtype=bool)
    _, diagnostics = compute_sv_distance_mixture_oe(
        matrix,
        valid,
        [(1, 4, 0.8), (4, 8, 0.7)],
        max_sv_hops=3,
        return_diagnostics=True,
    )
    assert diagnostics["effective_distance"][1, 8] == 2
    assert diagnostics["sv_hops"][1, 8] == 2
    assert diagnostics["mixture_weight"][1, 8] == pytest.approx(0.7)


def test_uniform_loading_initialization_is_supported():
    rng = np.random.default_rng(12)
    y = rng.normal(size=(80, 4))
    model = CNVAwareSSM(
        model_type="ar1", n_pcs=4, loading_init="uniform",
        max_iter=3, verbose=False,
    )
    result = model.fit(
        y=y, c=np.zeros(80), valid_mask=np.ones(80, dtype=bool)
    )
    assert result["H"].shape == (4, 1)
    assert np.all(np.isfinite(result["H"]))


def test_phasing_loading_initialization_requires_track():
    model = CNVAwareSSM(n_pcs=2, loading_init="phasing", verbose=False)
    with pytest.raises(ValueError, match="requires prior_phasing_track"):
        model.fit(
            y=np.ones((20, 2)), c=np.zeros(20),
            valid_mask=np.ones(20, dtype=bool),
        )


def test_fixed_cnv_effect_is_not_reestimated_by_em():
    rng = np.random.default_rng(13)
    c = np.repeat([0.0, 1.0], 60)
    y = np.column_stack([2.0 * c, -0.5 * c]) + rng.normal(0, 0.1, (120, 2))
    expected = np.array([stats.linregress(c, y[:, k]).slope for k in range(2)])
    model = CNVAwareSSM(
        n_pcs=2, max_iter=10, freeze_cnv_effect=True, verbose=False
    )
    result = model.fit(y=y, c=c, valid_mask=np.ones(120, dtype=bool))
    np.testing.assert_allclose(result["G"][:, 0], expected)


def test_sv_graph_transition_couples_paired_endpoints():
    means = np.zeros((20, 1))
    means[3, 0], means[16, 0] = 2.0, -2.0
    covs = np.repeat(np.eye(1)[None, :, :] * 0.2, 20, axis=0)
    model = CNVAwareSSM(n_pcs=1, sv_edge_strength=5.0, verbose=False)
    updated = model._apply_sv_graph_transitions(
        means, covs, np.array([[0.95]]), np.array([[0.1]]), [(3, 16)]
    )
    before = abs(means[16, 0] - 0.95 * means[3, 0])
    after = abs(updated[16, 0] - 0.95 * updated[3, 0])
    assert after < before


def test_contact_graph_uses_mutual_distal_neighbors_and_threshold():
    corr = np.eye(7)
    corr[0, 4] = corr[4, 0] = 0.90
    corr[0, 5] = corr[5, 0] = 0.80
    corr[1, 5] = corr[5, 1] = 0.85
    corr[2, 6] = corr[6, 2] = 0.20  # below threshold

    edges = build_contact_knn_graph(
        corr, np.ones(7, dtype=bool), k=1, min_distance_bins=3,
        min_correlation=0.3, mutual=True,
    )
    pairs = {(edge[0], edge[1]) for edge in edges}
    assert pairs == {(0, 4), (1, 5)}
    assert all(np.isfinite(edge[2]) and edge[2] > 0 for edge in edges)


def test_contact_graph_update_pulls_correlated_distal_states_together():
    means = np.zeros((12, 1))
    means[1, 0], means[10, 0] = -2.0, 2.0
    covs = np.repeat(np.eye(1)[None, :, :] * 0.2, 12, axis=0)
    model = CNVAwareSSM(
        n_pcs=1, contact_graph_strength=2.0, verbose=False,
    )
    updated = model._apply_contact_graph(
        means, covs, np.array([[0.1]]), [(1, 10, 1.0)]
    )
    assert abs(updated[1, 0] - updated[10, 0]) < 4.0


def test_signed_contact_graph_enforces_opposite_states_for_negative_edge():
    means = np.zeros((12, 1))
    means[1, 0], means[10, 0] = 2.0, 2.0
    covs = np.repeat(np.eye(1)[None, :, :] * 0.2, 12, axis=0)
    model = CNVAwareSSM(
        n_pcs=1, contact_graph_strength=2.0, verbose=False,
    )
    updated = model._apply_contact_graph(
        means, covs, np.array([[0.1]]), [(1, 10, 1.0, -1.0)]
    )
    assert abs(updated[1, 0] + updated[10, 0]) < 4.0


def test_two_scale_returns_global_and_local_states():
    rng = np.random.default_rng(7)
    T = 160
    global_state = np.sin(np.linspace(0, 4 * np.pi, T))
    local_state = np.zeros(T)
    local_state[70:90] = 2.0
    y = np.column_stack([
        global_state + rng.normal(0, 0.08, T),
        local_state + rng.normal(0, 0.08, T),
        rng.normal(0, 0.15, T),
    ])
    breakpoints = np.zeros(T, dtype=bool)
    breakpoints[[0, 70, 90]] = True

    model = CNVAwareSSM(
        model_type="two_scale", n_pcs=3, max_iter=80, tol=1e-4, verbose=False
    )
    result = model.fit(
        y=y,
        c=np.zeros(T),
        valid_mask=np.ones(T, dtype=bool),
        breakpoints=breakpoints,
    )

    assert result["local_artifact_score"] is not None
    assert result["H"].shape == (3, 2)
    assert result["F"][0, 0] > result["F"][1, 1]
    assert np.corrcoef(global_state, result["state_score"])[0, 1] > 0.8


def test_breakpoints_bridge_short_missing_runs_and_ignore_cnv_by_default():
    import pandas as pd

    bins = pd.DataFrame({"start": np.arange(8) * 50_000,
                         "end": (np.arange(8) + 1) * 50_000})
    valid = np.array([True, True, False, True, False, False, False, True])
    cnv = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=float)

    bp = identify_transition_breakpoints(bins, valid, cnv, min_gap_bins=3)
    assert not bp[3]                 # bridge the isolated missing bin
    assert bp[7]                     # restart after three missing bins

    bp_cnv = identify_transition_breakpoints(
        bins, np.ones(8, dtype=bool), cnv,
        min_gap_bins=3, decouple_cnv_jumps=True,
    )
    assert bp_cnv[3]


def test_eaglec2_sv_endpoints_become_ar_boundaries(tmp_path):
    import pandas as pd

    sv = tmp_path / "calls.txt"
    sv.write_text(
        "chrom1\tpos1\tchrom2\tpos2\n"
        "chr18\t100000\tchr18\t300000\n"
        "chr18\t200000\tchr19\t400000\n"
    )
    bins = pd.DataFrame({
        "chrom": ["chr18"] * 10,
        "start": np.arange(10) * 50000,
        "end": (np.arange(10) + 1) * 50000,
    })
    bp, edges = load_sv_endpoint_breakpoints(str(sv), "chr18", bins)
    assert edges == [(2, 6)]
    assert np.flatnonzero(bp).tolist() == [2, 6]


def test_sv_endpoint_inside_bin_maps_to_containing_bin(tmp_path):
    import pandas as pd

    sv = tmp_path / "calls.txt"
    sv.write_text(
        "chrom1\tpos1\tchrom2\tpos2\n"
        "chr18\t120000\tchr18\t389999\n"
    )
    bins = pd.DataFrame({
        "chrom": ["chr18"] * 10,
        "start": np.arange(10) * 50000,
        "end": (np.arange(10) + 1) * 50000,
    })
    bp, edges = load_sv_endpoint_breakpoints(str(sv), "chr18", bins)
    assert edges == [(2, 7)]
    assert np.flatnonzero(bp).tolist() == [2, 7]


def test_pearson_uses_consistent_population_standardization():
    from cnv_latent_ssm.features import compute_interaction_profile_correlation

    matrix = np.array([[1., 2., 4.], [2., 1., 0.], [4., 0., 2.]])
    observed = compute_interaction_profile_correlation(
        matrix, np.ones(3, dtype=bool), log_transform=False,
        winsorize_quantile=1.0,
    )
    expected = np.corrcoef(matrix)
    np.testing.assert_allclose(observed, expected)


def test_sv_edges_are_weighted_by_endpoint_copy_dosage():
    weighted, table = weight_sv_edges_by_cnv(
        [(0, 1), (2, 3)], np.array([1.0, 1.0, 4.0, 4.0])
    )
    assert weighted[1][2] > weighted[0][2]
    np.testing.assert_allclose(table.sv_cn_dosage_proxy, [1.0, 4.0])


def test_raw_values_use_balancing_weight_mask():
    import pandas as pd

    matrix = np.array([[2.0, 1.0], [1.0, 2.0]])
    bins = pd.DataFrame({"weight": [1.0, np.nan]})

    masked_valid = identify_valid_bins(matrix, bins_df=bins)
    unmasked_valid = identify_valid_bins(
        matrix, bins_df=bins, require_balancing_weight=False
    )

    np.testing.assert_array_equal(masked_valid, [True, False])
    np.testing.assert_array_equal(unmasked_valid, [True, True])


def test_cnvkit_log2_segments_are_not_converted_through_ploidy(tmp_path):
    import pandas as pd

    cns = tmp_path / "sample.cns"
    cns.write_text(
        "chromosome\tstart\tend\tgene\tlog2\n"
        "chr18\t0\t100000\t-\t-0.25\n"
        "chr18\t100000\t200000\t-\t0.40\n"
    )
    bins = pd.DataFrame({
        "chrom": ["chr18"] * 4,
        "start": [0, 50000, 100000, 150000],
        "end": [50000, 100000, 150000, 200000],
    })
    deviation, ratio, deleted = load_external_cnv(
        str(cns), "chr18", bins, value_type="log2_ratio"
    )
    np.testing.assert_allclose(deviation, [-0.25, -0.25, 0.40, 0.40])
    np.testing.assert_allclose(ratio, np.exp2(deviation))
    assert not deleted.any()


def test_cnvkit_log2_input_can_use_linear_relative_cn_effect(tmp_path):
    import pandas as pd

    cns = tmp_path / "sample.cns"
    cns.write_text(
        "chromosome\tstart\tend\tgene\tlog2\n"
        "chr18\t0\t50000\t-\t-1.0\n"
        "chr18\t50000\t100000\t-\t1.0\n"
    )
    bins = pd.DataFrame({
        "chrom": ["chr18", "chr18"], "start": [0, 50000],
        "end": [50000, 100000],
    })
    deviation, ratio, deleted, meta = get_cnv_track(
        "external", "chr18", bins, external_cnv_file=str(cns),
        value_type="log2_ratio", effect_scale="linear",
    )
    np.testing.assert_allclose(ratio, [0.5, 2.0])
    np.testing.assert_allclose(deviation, [-0.5, 1.0])
    assert meta["effect_scale"] == "linear"


def test_calder_style_rank_score_preserves_mask_and_range():
    score = calder_style_rank_score(np.array([10.0, np.nan, 30.0, 20.0]))
    np.testing.assert_allclose(score[[0, 2, 3]], [-1.0, 1.0, 0.0])
    assert np.isnan(score[1])
