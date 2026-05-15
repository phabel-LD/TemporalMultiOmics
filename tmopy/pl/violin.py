"""Violin plots for validation results."""

import matplotlib.pyplot as plt
import numpy as np


def plot_violin_validation(
    target_values,
    background_values,
    target_label="Target",
    background_label="Background",
    title="Validation result",
    ylabel="Δτ shift / ADS",
    save=None,
    show=True,
    figsize=(6, 6),
):
    """
    Violin plot comparing two groups (e.g., target vs background).

    Parameters
    ----------
    target_values, background_values : array-like
        Values for the two groups.
    target_label, background_label : str
        Labels for the two groups.
    title, ylabel : str
        Plot title and y‑axis label.
    save : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the figure.
    figsize : tuple, default=(6,6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    data = [target_values, background_values]
    labels = [target_label, background_label]

    fig, ax = plt.subplots(figsize=figsize)
    parts = ax.violinplot(data, positions=[1, 2], showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['lightblue', 'lightcoral'][i])
        pc.set_alpha(0.7)
    # Add jittered points
    for i, vals in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.3, s=5, color='black')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig