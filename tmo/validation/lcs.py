"""Lag Concordance Score (LCS) validation.

LCS = Spearman correlation between learned Δτ̂_g (per gene) and
CCF-derived Δτ_g^CCF. This is a key metric for evaluating whether
the model has learned biologically meaningful regulatory lags.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr


def compute_lcs(
    predicted_lags: Dict[int, float],
    target_lags: Dict[int, float],
) -> Tuple[float, float]:
    """Compute Lag Concordance Score (LCS) from per-gene predictions.

    Parameters
    ----------
    predicted_lags : dict
        Mapping from gene ID to predicted Δτ̂_g (from model).
    target_lags : dict
        Mapping from gene ID to CCF-derived Δτ_g^CCF.

    Returns
    -------
    lcs : float
        Spearman correlation coefficient.
    p_value : float
        Two-sided p-value.
    """
    # Find common genes
    common_ids = set(predicted_lags.keys()) & set(target_lags.keys())
    if len(common_ids) < 2:
        return np.nan, np.nan

    pred_list = [predicted_lags[g] for g in common_ids]
    target_list = [target_lags[g] for g in common_ids]

    corr, p_val = spearmanr(pred_list, target_list)
    return corr, p_val


def compute_lcs_from_arrays(
    predicted_lags: np.ndarray,
    target_lags: np.ndarray,
    gene_ids: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute LCS from aligned arrays.

    Parameters
    ----------
    predicted_lags : np.ndarray, shape (n_genes,)
        Predicted lags.
    target_lags : np.ndarray, shape (n_genes,)
        Target lags (CCF-derived).
    gene_ids : np.ndarray, optional
        Gene IDs for filtering (e.g., to remove genes with NaN targets).

    Returns
    -------
    lcs : float
        Spearman correlation.
    p_value : float
        Two-sided p-value.
    """
    # Remove NaN targets
    valid = ~np.isnan(target_lags)
    if gene_ids is not None:
        valid = valid & ~np.isnan(gene_ids)
    if np.sum(valid) < 2:
        return np.nan, np.nan

    pred_valid = predicted_lags[valid]
    target_valid = target_lags[valid]

    corr, p_val = spearmanr(pred_valid, target_valid)
    return corr, p_val


class LCSLogger:
    """Utility for tracking LCS during training."""

    def __init__(self):
        self.history: List[Tuple[int, float]] = []  # (step, lcs)

    def log(self, step: int, lcs: float):
        """Record LCS at a given step."""
        self.history.append((step, lcs))

    def get_best(self) -> Tuple[int, float]:
        """Return step and value of best LCS."""
        if not self.history:
            return -1, -np.inf
        best_step, best_lcs = max(self.history, key=lambda x: x[1])
        return best_step, best_lcs

    def get_latest(self) -> float:
        """Return most recent LCS."""
        if not self.history:
            return np.nan
        return self.history[-1][1]