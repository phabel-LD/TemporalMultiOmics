"""Perturbation-induced lag shift (δΔτ) validation.

δΔτ_g^(k) = Δτ_g^CCF(perturbed) - Δτ_g^CCF(unperturbed)

This metric tests whether genes that are known targets of a perturbation
(e.g., TF knockout) show larger |δΔτ| than non-target genes.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def compute_lag_shift(
    delta_tau_perturbed: Dict[int, float],
    delta_tau_unperturbed: Dict[int, float],
) -> Dict[int, float]:
    """Compute δΔτ for each gene.

    Parameters
    ----------
    delta_tau_perturbed : dict
        Gene ID -> Δτ in perturbed condition.
    delta_tau_unperturbed : dict
        Gene ID -> Δτ in unperturbed (control) condition.

    Returns
    -------
    dict
        Gene ID -> δΔτ = Δτ_perturbed - Δτ_unperturbed.
    """
    common_ids = set(delta_tau_perturbed.keys()) & set(delta_tau_unperturbed.keys())
    delta = {}
    for g in common_ids:
        delta[g] = delta_tau_perturbed[g] - delta_tau_unperturbed[g]
    return delta


def compute_lag_shift_from_arrays(
    delta_tau_perturbed: np.ndarray,
    delta_tau_unperturbed: np.ndarray,
    gene_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute δΔτ from aligned arrays.

    Parameters
    ----------
    delta_tau_perturbed : np.ndarray, shape (n_genes,)
        Lags in perturbed condition.
    delta_tau_unperturbed : np.ndarray, shape (n_genes,)
        Lags in unperturbed condition.
    gene_ids : np.ndarray, shape (n_genes,)
        Gene IDs.

    Returns
    -------
    delta_tau_shift : np.ndarray
        δΔτ for each gene (NaN where either input is NaN).
    valid_gene_ids : np.ndarray
        Gene IDs for non-NaN entries.
    """
    valid = ~(np.isnan(delta_tau_perturbed) | np.isnan(delta_tau_unperturbed))
    delta = delta_tau_perturbed[valid] - delta_tau_unperturbed[valid]
    return delta, gene_ids[valid]


def test_lag_shift_significance(
    delta_shift_target: np.ndarray,
    delta_shift_background: np.ndarray,
    alternative: str = 'greater',
) -> Tuple[float, float]:
    """Test whether target genes have larger |δΔτ| than background.

    Parameters
    ----------
    delta_shift_target : np.ndarray
        Absolute lag shifts for target genes.
    delta_shift_background : np.ndarray
        Absolute lag shifts for non-target genes.
    alternative : str, default='greater'
        'greater' tests if target > background; 'two-sided' for difference.

    Returns
    -------
    statistic : float
        Mann-Whitney U statistic.
    p_value : float
        One-sided or two-sided p-value.
    """
    stat, p = mannwhitneyu(
        delta_shift_target,
        delta_shift_background,
        alternative=alternative,
    )
    return stat, p


def compute_auc_for_lag_shift(
    delta_shift_abs: np.ndarray,
    is_target: np.ndarray,
) -> float:
    """Compute AUROC for distinguishing target vs. non-target based on |δΔτ|.

    Parameters
    ----------
    delta_shift_abs : np.ndarray, shape (n_genes,)
        Absolute lag shifts.
    is_target : np.ndarray, bool, shape (n_genes,)
        True for target genes.

    Returns
    -------
    auc : float
        Area under ROC curve.
    """
    if len(np.unique(is_target)) < 2:
        return np.nan
    return roc_auc_score(is_target, delta_shift_abs)


class DeltaTauAnalyzer:
    """Analyzer for perturbation-induced lag shifts across multiple conditions."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}  # perturbation_id -> result dict

    def add_perturbation(
        self,
        pert_id: str,
        delta_shift_target: np.ndarray,
        delta_shift_background: np.ndarray,
        target_genes: List[int],
    ):
        """Store analysis results for a perturbation.

        Parameters
        ----------
        pert_id : str
            Identifier for the perturbation (e.g., 'GATA1_KO').
        delta_shift_target : np.ndarray
            |δΔτ| for target genes.
        delta_shift_background : np.ndarray
            |δΔτ| for background genes.
        target_genes : list
            List of target gene IDs for reference.
        """
        stat, p = test_lag_shift_significance(
            delta_shift_target,
            delta_shift_background,
            alternative='greater',
        )
        median_target = np.median(delta_shift_target)
        median_background = np.median(delta_shift_background)
        self.results[pert_id] = {
            'statistic': stat,
            'p_value': p,
            'median_target': median_target,
            'median_background': median_background,
            'n_target': len(delta_shift_target),
            'n_background': len(delta_shift_background),
            'target_genes': target_genes,
        }

    def get_summary(self) -> dict:
        """Return summary statistics across all perturbations."""
        summary = {
            'significant_perturbations': [],
            'mean_p_value': np.nan,
            'mean_effect_size': np.nan,
        }
        p_vals = []
        effect_sizes = []
        for pert_id, res in self.results.items():
            p_vals.append(res['p_value'])
            effect = res['median_target'] - res['median_background']
            effect_sizes.append(effect)
            if res['p_value'] < 0.05:
                summary['significant_perturbations'].append(pert_id)
        if p_vals:
            summary['mean_p_value'] = np.mean(p_vals)
            summary['mean_effect_size'] = np.mean(effect_sizes)
        return summary