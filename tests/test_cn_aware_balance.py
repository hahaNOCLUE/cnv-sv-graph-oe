import numpy as np

from cnv_latent_ssm.cn_aware_balance import fit_cn_aware_balance


def test_cn_aware_balance_equalizes_per_copy_marginals():
    cn = np.array([1., 1.5, 2., 3.])
    truth = np.array([.8, 1.1, 1.25, .9])
    base = np.outer(cn, cn)
    observed = base * np.outer(truth, truth)
    np.fill_diagonal(observed, 0.)
    fit = fit_cn_aware_balance(observed, cn, np.ones(4, bool),
                               bias_min=.2, bias_max=5., tolerance=1e-7)
    corrected = observed / np.outer(fit.bias, fit.bias)
    per_copy = corrected.sum(axis=1) / cn
    np.testing.assert_allclose(per_copy, np.repeat(per_copy.mean(), 4), rtol=2e-3)
