"""Utilities for Perturb‑seq benchmark (Norman et al., combinatorial generalization)."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import r2_score


def compute_combinatorial_perturbation_accuracy(
    pred_expr: np.ndarray,
    true_expr: np.ndarray,
    gene_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Compute prediction accuracy for combinatorial perturbations.

    Parameters
    ----------
    pred_expr : np.ndarray, shape (n_perturbations, n_genes)
        Predicted expression for each combinatorial perturbation.
    true_expr : np.ndarray, same shape
        True observed expression.
    gene_names : list, optional
        For per‑gene metrics.

    Returns
    -------
    dict
        - 'r2_all': R² across all genes and perturbations
        - 'r2_per_perturbation': mean R² per perturbation
        - 'r2_per_gene': mean R² per gene (if gene_names provided)
    """
    r2_all = r2_score(true_expr.flatten(), pred_expr.flatten())
    per_pert = [r2_score(true_expr[i], pred_expr[i]) for i in range(true_expr.shape[0])]
    mean_per_pert = np.mean(per_pert)
    result = {'r2_all': r2_all, 'r2_per_perturbation': mean_per_pert}

    if gene_names is not None:
        per_gene = [r2_score(true_expr[:, j], pred_expr[:, j]) for j in range(true_expr.shape[1])]
        result['r2_per_gene'] = np.mean(per_gene)
        result['per_gene_scores'] = dict(zip(gene_names, per_gene))

    return result


def load_norman_perturbseq_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load the Norman et al. Perturb‑seq dataset (placeholder).

    In practice, download from GEO and parse. This function simulates data.

    Returns
    -------
    train_X : sim; train_Y ; test_X ; test_Y ; gene_names
    """
    # Simulate: 100 perturbations, 2000 genes
    np.random.seed(42)
    n_pert = 100
    n_genes = 2000
    train_X = np.random.randn(n_pert // 2, n_genes)
    train_Y = train_X + 0.1 * np.random.randn(n_pert // 2, n_genes)
    test_X = np.random.randn(n_pert // 2, n_genes)
    test_Y = test_X + 0.1 * np.random.randn(n_pert // 2, n_genes)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    return train_X, train_Y, test_X, test_Y, gene_names