"""Attention Disruption Score (ADS) validation.

ADS_g^(k) = Σ_{ℓ ∈ L_k} | w_{ℓ→g}^{perturbed} - w_{ℓ→g}^{unperturbed} |

where:
- L_k: set of ATAC peaks associated with transcription factor k (e.g., peaks containing its motif)
- w_{ℓ→g}: attention weight from ATAC peak ℓ to RNA token of gene g (averaged over cells)

Higher ADS for known target genes indicates that the attention mechanism has learned
genuine regulatory relationships.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score


def compute_attention_weights(
    attention_matrices: List[torch.Tensor],
    atac_peak_indices: List[int],
    gene_indices: List[int],
) -> np.ndarray:
    """Extract attention weights from ATAC peaks to RNA genes.

    Parameters
    ----------
    attention_matrices : list of torch.Tensor
        List of attention weight tensors, each shape (batch, n_heads, seq_len_atac, seq_len_rna).
        Typically from multiple layers or averaged.
    atac_peak_indices : list of int
        Indices of ATAC peaks in the token sequence (positions of ATAC tokens).
    gene_indices : list of int
        Indices of RNA gene tokens in the token sequence.

    Returns
    -------
    weights : np.ndarray, shape (n_atac_peaks, n_genes)
        Average attention weight from each ATAC peak to each gene.
        Averaged over batch, heads, and layers.
    """
    # Stack and average over layers and heads
    # Assuming list of tensors from different layers
    all_attns = torch.stack(attention_matrices, dim=0)  # (n_layers, batch, heads, seqA, seqR)
    # Average over layers, batch, heads
    avg_attn = all_attns.mean(dim=(0, 1, 2))  # (seqA, seqR)
    # Extract relevant rows (ATAC peaks) and columns (genes)
    weights = avg_attn[atac_peak_indices, :][:, gene_indices]  # (n_peaks, n_genes)
    return weights.cpu().numpy()


def compute_ads_for_perturbation(
    weights_unperturbed: np.ndarray,
    weights_perturbed: np.ndarray,
    tf_peak_indices: List[int],
    target_gene_indices: Set[int],
    background_gene_indices: Optional[Set[int]] = None,
) -> Dict[str, float]:
    """Compute Attention Disruption Score for a single TF perturbation.

    Parameters
    ----------
    weights_unperturbed : np.ndarray, shape (n_peaks, n_genes)
        Attention weights in control condition.
    weights_perturbed : np.ndarray, shape (n_peaks, n_genes)
        Attention weights in perturbed condition.
    tf_peak_indices : list of int
        Indices of ATAC peaks associated with the perturbed TF (L_k).
    target_gene_indices : set of int
        Indices of genes known to be targets of the TF.
    background_gene_indices : set of int, optional
        Indices of background (non-target) genes. If None, use all genes not in target.

    Returns
    -------
    dict
        Contains:
        - ads_target: list of ADS values for target genes
        - ads_background: list of ADS values for background genes
        - mean_ads_target: float
        - mean_ads_background: float
        - auc: AUROC for distinguishing target vs background based on ADS
        - p_value: Mann-Whitney p-value (target > background)
    """
    n_peaks, n_genes = weights_unperturbed.shape

    # Compute ADS for each gene: sum over TF-associated peaks of absolute difference
    delta = np.abs(weights_perturbed - weights_unperturbed)  # (n_peaks, n_genes)
    ads_per_gene = delta[tf_peak_indices, :].sum(axis=0)  # (n_genes,)

    # Separate target and background
    if background_gene_indices is None:
        background_gene_indices = set(range(n_genes)) - target_gene_indices

    ads_target = [ads_per_gene[g] for g in target_gene_indices if g < n_genes]
    ads_background = [ads_per_gene[g] for g in background_gene_indices if g < n_genes]

    # Compute statistics
    mean_target = np.mean(ads_target) if ads_target else np.nan
    mean_background = np.mean(ads_background) if ads_background else np.nan

    # AUROC
    labels = np.array([1] * len(ads_target) + [0] * len(ads_background))
    scores = np.array(ads_target + ads_background)
    auc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else np.nan

    # Mann-Whitney test (one-sided: target > background)
    stat, p = mannwhitneyu(ads_target, ads_background, alternative='greater')

    return {
        'ads_target': ads_target,
        'ads_background': ads_background,
        'mean_ads_target': mean_target,
        'mean_ads_background': mean_background,
        'auc': auc,
        'statistic': stat,
        'p_value': p,
    }


class ADSAnalyzer:
    """Analyzer for Attention Disruption Scores across multiple perturbations."""

    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def add_perturbation(
        self,
        pert_id: str,
        weights_unperturbed: np.ndarray,
        weights_perturbed: np.ndarray,
        tf_peak_indices: List[int],
        target_gene_indices: Set[int],
        background_gene_indices: Optional[Set[int]] = None,
    ):
        """Compute and store ADS for a perturbation."""
        result = compute_ads_for_perturbation(
            weights_unperturbed,
            weights_perturbed,
            tf_peak_indices,
            target_gene_indices,
            background_gene_indices,
        )
        self.results[pert_id] = result

    def get_summary(self) -> dict:
        """Return summary across all perturbations."""
        aucs = []
        p_vals = []
        mean_targets = []
        mean_backgrounds = []
        for pert_id, res in self.results.items():
            if not np.isnan(res['auc']):
                aucs.append(res['auc'])
            p_vals.append(res['p_value'])
            mean_targets.append(res['mean_ads_target'])
            mean_backgrounds.append(res['mean_ads_background'])

        summary = {
            'mean_auc': np.mean(aucs) if aucs else np.nan,
            'mean_p_value': np.mean(p_vals) if p_vals else np.nan,
            'mean_ads_target': np.mean(mean_targets) if mean_targets else np.nan,
            'mean_ads_background': np.mean(mean_backgrounds) if mean_backgrounds else np.nan,
            'significant_perturbations': [
                pert_id for pert_id, res in self.results.items() if res['p_value'] < 0.05
            ],
        }
        return summary


def compute_motif_peak_mapping(
    peak_to_motif: Dict[int, List[str]],
    tf_name_to_motif: Dict[str, List[str]],
    tf_of_interest: str,
) -> List[int]:
    """Helper: get indices of peaks containing motif for a given TF.

    Parameters
    ----------
    peak_to_motif : dict
        Mapping from peak index to list of motif names found in that peak.
    tf_name_to_motif : dict
        Mapping from TF name to list of motif names (e.g., 'GATA1' -> ['GATA1', 'GATA2']).
    tf_of_interest : str
        Name of the transcription factor.

    Returns
    -------
    list of int
        Peak indices that contain any motif for this TF.
    """
    motifs = set(tf_name_to_motif.get(tf_of_interest, []))
    peak_indices = []
    for peak_idx, motif_list in peak_to_motif.items():
        if any(m in motifs for m in motif_list):
            peak_indices.append(peak_idx)
    return peak_indices