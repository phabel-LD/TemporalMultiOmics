"""Training curves (loss and LCS) from CSV file."""

import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curves(
    metrics_csv: str,
    save: str = None,
    show: bool = True,
    figsize: tuple = (10, 6),
):
    """
    Plot training loss and validation LCS over epochs.

    Parameters
    ----------
    metrics_csv : str
        Path to CSV file with columns 'epoch', 'loss', 'lcs'.
    save : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the figure.
    figsize : tuple, default=(10,6)
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    df = pd.read_csv(metrics_csv)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color='blue')
    ax1.plot(df['epoch'], df['loss'], 'b-', label='Loss')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel("LCS (validation)", color='red')
    lcs_data = df[['epoch', 'lcs']].dropna()
    ax2.plot(lcs_data['epoch'], lcs_data['lcs'], 'ro-', label='LCS')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Training curves")
    fig.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig