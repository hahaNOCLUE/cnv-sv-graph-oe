import numpy as np

from cnv_latent_ssm.graph_visibility import (
    build_graph_visibility_mask,
    fit_graph_visibility,
)


def test_graph_visibility_recovers_bounded_bin_effects():
    n = 8
    graph_expected = np.ones((n, n), float) * 20
    np.fill_diagonal(graph_expected, 0)
    q_true = np.array([.7, .8, .9, 1., 1.1, 1.2, 1.3, 1.4])
    scale_true = 3.0
    observed = scale_true * np.outer(q_true, q_true) * graph_expected
    fit_mask = np.ones((n, n), bool)
    np.fill_diagonal(fit_mask, False)
    result = fit_graph_visibility(
        observed, graph_expected, fit_mask, np.ones(n, bool),
        damping=.5, tolerance=1e-6, max_iterations=500,
    )
    normalized_truth = q_true / np.exp(np.median(np.log(q_true)))
    assert result.converged
    np.testing.assert_allclose(result.visibility, normalized_truth, rtol=2e-3)
    fitted = result.scale * np.outer(result.visibility, result.visibility)
    np.testing.assert_allclose(
        fitted[fit_mask] * graph_expected[fit_mask], observed[fit_mask],
        rtol=3e-3,
    )


def test_graph_visibility_mask_uses_graph_distance_and_exclusions():
    observed = np.ones((4, 4), float)
    expected = np.ones((4, 4), float)
    cis = np.ones((4, 4), float)
    distance = np.array([
        [0, 10, 30, 120],
        [10, 0, 60, 120],
        [30, 60, 0, 20],
        [120, 120, 20, 0],
    ], float)
    excluded = np.array([False, False, False, True])
    mask = build_graph_visibility_mask(
        observed, expected, cis, distance, np.ones(4, bool), 50_000,
        excluded_bins=excluded, min_distance_bp=500_000,
        max_distance_bp=5_000_000, enrichment_quantile=1,
    )
    assert mask[0, 1]  # 10 graph bins = 0.5 Mb
    assert mask[1, 2]  # 60 graph bins = 3 Mb
    assert not mask[0, 3]  # excluded endpoint bin
    assert not mask[0, 0]
