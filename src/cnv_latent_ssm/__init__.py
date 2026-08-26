"""CNV-aware Latent Chromatin State Model package."""

from .caller import run_cnv_latent_ssm
from .cnv import get_cnv_track, load_external_cnv
from .features import (
    compute_interaction_profile_correlation,
    compute_observed_over_expected,
    extract_chrom_matrix_and_bins,
    extract_pca_observation_features,
)
from .ssm import CNVAwareSSM, SSMResults

__version__ = "0.1.0"
__all__ = [
    "run_cnv_latent_ssm",
    "CNVAwareSSM",
    "SSMResults",
    "load_external_cnv",
    "get_cnv_track",
    "extract_chrom_matrix_and_bins",
    "compute_observed_over_expected",
    "compute_interaction_profile_correlation",
    "extract_pca_observation_features",
]
