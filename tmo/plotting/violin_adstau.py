"""Violin plots for perturbation validation (Figure 4).

Shows:
    - δΔτ (lag shift) distribution for target vs. background genes
    - ADS distribution for target vs. background genes
    - Optional: side-by-side comparisons for multiple perturbations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from scipy.stats import mannwhitneyu


def plot_lag_shift_violin(
    delta_tau_target: np.ndarray,
    delta_tau_background: np.ndarray,
    target_label: str = "Target genes",
    background_label: str = "Background genes",
    title: str = "Perturbation-induced lag shift (δΔτ)",
    ylabel: str = "|δΔτ| (pseudotime units)",
    figsize: Tuple[int, int] = (6, 8),
    show_pvalue: bool = True,
    pvalue_threshold: float = 0.05,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plot comparing |δΔτ| between target and background genes.

    Parameters
    ----------
    delta_tau_target, delta_tau_background : arrays
        Absolute lag shift values for target and background genes.
    target_label, background_label, title, ylabel, figsize.
    show_pvalue : bool, default=True
        Annotate plot with Mann-Whitney p-value.
    pvalue_threshold : float, default=0.05
        Threshold for significance annotation (e.g., '*' if p < 0.05).
    save_path : optional.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data
    data = [delta_tau_target, delta_tau_background]
    labels = [target_label, background_label]

    # Violin plot
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True,
                           widths=0.7, vert=True)
    # Color violins
    colors = ['lightblue', 'lightcoral']
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    # Add boxplot inside
    ax.boxplot(data, positions=[1, 2], widths=0.2, patch_artist=True,
               boxprops=dict(facecolor='white', alpha=0.7))

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add p-value annotation
    if show_pvalue:
        stat, p = mannwhitneyu(delta_tau_target, delta_tau_background, alternative='greater')
        p_text = f"p = {p:.2e}" if p < 0.001 else f"p = {p:.4f}"
        if p < pvalue_threshold:
            p_text = "* " + p_text
        # Position annotation
        y_max = max(np.max(delta_tau_target), np.max(delta_tau_background))
        ax.text(1.5, y_max * 0.95, p_text, ha='center', va='top', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_ads_violin(
    ads_target: np.ndarray,
    ads_background: np.ndarray,
    target_label: str = "Target genes",
    background_label: str = "Background genes",
    title: str = "Attention Disruption Score (ADS)",
    ylabel: str = "ADS",
    figsize: Tuple[int, int] = (6, 8),
    show_pvalue: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Violin plot comparing ADS between target and background genes."""
    return plot_lag_shift_violin(
        ads_target, ads_background,
        target_label, background_label,
        title, ylabel, figsize, show_pvalue, save_path=save_path,
    )


def plot_multi_perturbation_comparison(
    results: Dict[str, Dict[str, np.ndarray]],
    metric: str = 'ads',  # 'ads' or 'delta_tau'
    title: str = "Validation across perturbations",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side violin plots for multiple perturbations.

    Parameters
    ----------
    results : dict
        Mapping from perturbation name to dict with keys:
        - 'target': array of values for target genes
        - 'background': array of values for background genes
    metric : str, 'ads' or 'delta_tau'
        Which metric to plot.
    title, figsize, save_path.

    Returns
    -------
    plt.Figure
    """
    pert_names = list(results.keys())
    n_perts = len(pert_names)

    # Prepare positions: for each perturbation, two violins (target, background)
    positions_target = np.arange(1, 2 * n_perts, 2)
    positions_background = np.arange(2, 2 * n_perts, 2)

    fig, ax = plt.subplots(figsize=figsize)

    # Collect data for each perturbation
    target_vals = []
    background_vals = []
    for pert in pert_names:
        target_vals.append(results[pert]['target'])
        background_vals.append(results[pert]['background'])

    # Violin plots (alternating)
    all_data = []
    all_positions = []
    for i, (t, b) in enumerate(zip(target_vals, background_vals)):
        all_data.append(t)
        all_positions.append(positions_target[i])
        all_data.append(b)
        all_positions.append(positions_background[i])

    parts = ax.violinplot(all_data, positions=all_positions, showmeans=True,
                          showmedians=True, widths=0.7)
    # Color: target=blue, background=red
    colors = ['lightblue', 'lightcoral'] * n_perts
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Add boxplots
    ax.boxplot(all_data, positions=all_positions, widths=0.2, patch_artist=True,
               boxprops=dict(facecolor='white', alpha=0.7))

    # Set xticks: label pairs
    xtick_labels = []
    for pert in pert_names:
        xtick_labels.append(f"{pert}\nTarget")
        xtick_labels.append(f"{pert}\nBackground")
    ax.set_xticks(all_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
    ax.set_ylabel("ADS" if metric == 'ads' else "|δΔτ|")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig