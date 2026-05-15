"""Validation metrics: LCS, δΔτ, ADS."""

from .lcs import compute_lcs, compute_lcs_from_arrays, LCSLogger
from .delta_tau import (
    compute_lag_shift,
    compute_lag_shift_from_arrays,
    test_lag_shift_significance,
    compute_auc_for_lag_shift,
    DeltaTauAnalyzer,
)
from .ads import (
    compute_attention_weights,
    compute_ads_for_perturbation,
    ADSAnalyzer,
    compute_motif_peak_mapping,
)

__all__ = [
    "compute_lcs",
    "compute_lcs_from_arrays",
    "LCSLogger",
    "compute_lag_shift",
    "compute_lag_shift_from_arrays",
    "test_lag_shift_significance",
    "compute_auc_for_lag_shift",
    "DeltaTauAnalyzer",
    "compute_attention_weights",
    "compute_ads_for_perturbation",
    "ADSAnalyzer",
    "compute_motif_peak_mapping",
]