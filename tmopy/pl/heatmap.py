"""Δτ heatmap (Figure 1)."""

import matplotlib.pyplot as plt
from tmo.plotting.heatmap import plot_delta_tau_heatmap


def plot_delta_tau(
    delta_tau_matrix,
    pseudotime_grid,
    row_labels,
    title="Regulatory lag Δτ(τ)",
    save=None,
    show=True,
    **kwargs,
):
    """
    Plot Δτ heatmap.

    Parameters
    ----------
    delta_tau_matrix : np.ndarray, shape (n_rows, n_windows)
        Δτ values (returned by evaluate_asymmetric).
    pseudotime_grid : np.ndarray, shape (n_windows,)
        Pseudotime bin centres.
    row_labels : list of str
        Labels for each row (component or cluster names).
    title : str, default="Regulatory lag Δτ(τ)"
        Plot title.
    save : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the figure.
    **kwargs : passed to plot_delta_tau_heatmap (vmin, vmax, cmap, etc.).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plot_delta_tau_heatmap(
        delta_tau_matrix,
        pseudotime_grid,
        row_labels,
        title=title,
        save_path=save,
        **kwargs,
    )
    if show:
        plt.show()
    return fig