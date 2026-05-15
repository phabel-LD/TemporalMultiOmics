"""Attention offset plots (Figure 2).

Shows how attention weights depend on the temporal offset (τ_RNA - τ_ATAC)
for different gene classes (early priming, late response, neuronal).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from scipy.stats import gaussian_kde


def plot_attention_vs_offset(
    offsets: np.ndarray,
    attention_weights: np.ndarray,
    gene_class_labels: List[str],
    gene_class_colors: Optional[List[str]] = None,
    title: str = "Attention weight vs. temporal offset",
    xlabel: str = "τ_RNA - τ_ATAC (pseudotime units)",
    ylabel: str = "Attention weight",
    figsize: Tuple[int, int] = (10, 6),
    alpha: float = 0.5,
    smooth_kde: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Scatter or density plot of attention weights vs. offset.

    Parameters
    ----------
    offsets : np.ndarray, shape (n_pairs,)
        Temporal offset (τ_RNA - τ_ATAC) for each attended pair.
    attention_weights : np.ndarray, shape (n_pairs,)
        Corresponding attention weights (averaged over heads/layers).
    gene_class_labels : list of str, length n_pairs
        Class label for each point (e.g., 'early_priming', 'late_response').
    gene_class_colors : list of str, optional
        Colors for each unique class.
    title, xlabel, ylabel, figsize, alpha, smooth_kde, save_path.

    Returns
    -------
    plt.Figure
    """
    unique_classes = list(set(gene_class_labels))
    if gene_class_colors is None:
        palette = sns.color_palette("husl", len(unique_classes))
        gene_class_colors = {cls: palette[i] for i, cls in enumerate(unique_classes)}
    else:
        gene_class_colors = dict(zip(unique_classes, gene_class_colors))

    fig, ax = plt.subplots(figsize=figsize)

    if smooth_kde:
        # Plot KDE for each class
        for cls in unique_classes:
            mask = np.array(gene_class_labels) == cls
            if np.sum(mask) < 2:
                continue
            # Compute KDE
            xy = np.vstack([offsets[mask], attention_weights[mask]])
            try:
                kde = gaussian_kde(xy)
                # Create grid
                x_grid = np.linspace(offsets.min(), offsets.max(), 100)
                y_grid = np.linspace(attention_weights.min(), attention_weights.max(), 100)
                X, Y = np.meshgrid(x_grid, y_grid)
                Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
                ax.contourf(X, Y, Z, alpha=alpha, cmap=plt.cm.viridis)
            except:
                ax.scatter(offsets[mask], attention_weights[mask], alpha=alpha,
                           s=1, c=gene_class_colors[cls], label=cls)
    else:
        # Scatter plot
        for cls in unique_classes:
            mask = np.array(gene_class_labels) == cls
            ax.scatter(offsets[mask], attention_weights[mask],
                       alpha=alpha, s=5, c=gene_class_colors[cls], label=cls)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ridge_attention_by_gene_class(
    offsets_per_class: Dict[str, np.ndarray],
    attention_weights_per_class: Dict[str, np.ndarray],
    title: str = "Attention weight distribution by gene class",
    xlabel: str = "τ_RNA - τ_ATAC (pseudotime units)",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Ridge plot (joyplot) of attention weight distributions across offsets.

    Parameters
    ----------
    offsets_per_class : dict
        Class name -> array of offsets for that class.
    attention_weights_per_class : dict
        Class name -> array of attention weights (same length as offsets).
    title, xlabel, figsize, save_path.

    Returns
    -------
    plt.Figure
    """
    from scipy.interpolate import interp1d

    fig, ax = plt.subplots(figsize=figsize)
    classes = list(offsets_per_class.keys())
    # Create offset bins
    offset_bins = np.linspace(-0.4, 0.4, 50)
    offset_centers = (offset_bins[:-1] + offset_bins[1:]) / 2

    y_offset = 0
    y_spacing = 1.0
    colors = sns.color_palette("husl", len(classes))

    for i, cls in enumerate(classes):
        offsets = offsets_per_class[cls]
        attn = attention_weights_per_class[cls]
        # Bin offsets and compute mean attention per bin
        digitized = np.digitize(offsets, offset_bins)
        bin_means = []
        for b in range(1, len(offset_bins)):
            mask = digitized == b
            if np.any(mask):
                bin_means.append(np.mean(attn[mask]))
            else:
                bin_means.append(0)
        # Normalize
        bin_means = np.array(bin_means)
        if bin_means.max() > 0:
            bin_means = bin_means / bin_means.max()
        # Plot as filled area
        ax.fill_between(offset_centers, y_offset, y_offset + bin_means * y_spacing,
                         alpha=0.6, color=colors[i], label=cls)
        y_offset += y_spacing

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Gene class (offset)")
    ax.set_title(title)
    ax.set_yticks(np.arange(len(classes)) * y_spacing + y_spacing/2)
    ax.set_yticklabels(classes)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig