import numpy as np

from cnv_latent_ssm.graph_visibility import (
    build_graph_visibility_mask,
    fit_graph_visibility,
    split_visibility_mask,
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


def test_graph_visibility_mask_requires_dominant_band_exposure():
    observed = np.ones((4, 4), float)
    expected = np.ones((4, 4), float)
    band = np.ones((4, 4), float)
    band[0, 2] = band[2, 0] = .2
    excluded = np.array([False, False, False, True])
    mask = build_graph_visibility_mask(
        observed, expected, band, np.ones(4, bool),
        excluded_bins=excluded, minimum_band_fraction=.9,
        enrichment_quantile=1,
    )
    assert mask[0, 1]
    assert not mask[0, 2]
    assert mask[1, 2]
    assert not mask[0, 3]  # excluded endpoint bin
    assert not mask[0, 0]

    train, holdout = split_visibility_mask(mask, .2)
    assert not np.any(train & holdout)
    np.testing.assert_array_equal(train | holdout, mask)
    np.testing.assert_array_equal(train, train.T)
    np.testing.assert_array_equal(holdout, holdout.T)
