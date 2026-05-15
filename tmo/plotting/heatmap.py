"""Plotting utilities for TMO results.

This module provides functions to generate publication-ready figures.
This file focuses on heatmaps for Δτ surfaces (Figure 1).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
import pandas as pd


def plot_delta_tau_heatmap(
    delta_tau_matrix: np.ndarray,
    pseudotime_grid: np.ndarray,
    gene_names: List[str],
    title: str = "Regulatory lag Δτ(τ)",
    xlabel: str = "Pseudotime",
    ylabel: str = "Genes",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
    show_colorbar: bool = True,
    cbar_label: str = "Δτ (pseudotime units)",
    mask_threshold: Optional[float] = None,
) -> plt.Figure:
    """Plot signed Δτ surface as a heatmap.

    Parameters
    ----------
    delta_tau_matrix : np.ndarray, shape (n_genes, n_windows)
        Signed lag values (can be NaN for sparse windows).
    pseudotime_grid : np.ndarray, shape (n_windows,)
        Pseudotime coordinates for the x-axis.
    gene_names : list of str, length n_genes
        Gene names for the y-axis.
    title : str, default="Regulatory lag Δτ(τ)"
        Plot title.
    xlabel, ylabel : str
        Axis labels.
    cmap : str, default="RdBu_r"
        Colormap (diverging, good for signed values).
    center : float, default=0.0
        Colorbar center (useful for diverging colormap).
    figsize : tuple, default=(12,8)
        Figure size in inches.
    save_path : str, optional
        If provided, save figure to this path.
    show_colorbar : bool, default=True
        Whether to display colorbar.
    cbar_label : str, default="Δτ (pseudotime units)"
        Label for colorbar.
    mask_threshold : float, optional
        If provided, mask (whiten) values with absolute value below threshold.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data for heatmap
    data = delta_tau_matrix.copy()
    
    # Optionally mask small values
    if mask_threshold is not None:
        data = np.ma.masked_where(np.abs(data) < mask_threshold, data)
    
    # Create heatmap
    im = ax.imshow(
        data,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=[pseudotime_grid.min(), pseudotime_grid.max(), 0, len(gene_names)],
        origin='lower',
        vmin=-np.nanmax(np.abs(data)) if center == 0 else None,
        vmax=np.nanmax(np.abs(data)) if center == 0 else None,
    )
    
    # Set colormap center to zero for diverging maps
    if center == 0 and cmap in ['RdBu_r', 'coolwarm', 'seismic']:
        norm = plt.Normalize(vmin=-np.nanmax(np.abs(data)), vmax=np.nanmax(np.abs(data)))
        im.set_norm(norm)
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Set y-ticks (gene names) - show every ~20 genes if too many
    n_genes = len(gene_names)
    if n_genes <= 50:
        ax.set_yticks(np.arange(n_genes))
        ax.set_yticklabels(gene_names, fontsize=8)
    else:
        step = max(1, n_genes // 20)
        ax.set_yticks(np.arange(0, n_genes, step))
        ax.set_yticklabels([gene_names[i] for i in range(0, n_genes, step)], fontsize=8)
    
    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_delta_tau_with_trajectory(
    delta_tau_matrix: np.ndarray,
    pseudotime_grid: np.ndarray,
    gene_names: List[str],
    trajectory_stages: List[Tuple[float, str]],
    title: str = "Regulatory lag along differentiation",
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Δτ heatmap with annotated trajectory stages."""
    fig = plot_delta_tau_heatmap(
        delta_tau_matrix, pseudotime_grid, gene_names,
        title=title, figsize=figsize, save_path=None, show_colorbar=True,
    )
    ax = fig.axes[0]
    y_max = len(gene_names)
    for cutoff, name in trajectory_stages:
        ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(cutoff + 0.01, y_max * 0.95, name, rotation=90, verticalalignment='top', fontsize=10)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig


def plot_gene_signature_heatmaps(
    delta_tau_matrix: np.ndarray,
    pseudotime_grid: np.ndarray,
    gene_groups: Dict[str, List[int]],
    group_names: List[str],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot heatmaps for specific gene groups (e.g., early vs late genes)."""
    n_groups = len(group_names)
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize, sharex=True)
    if n_groups == 1:
        axes = [axes]
    
    for ax, group_name in zip(axes, group_names):
        gene_idx = gene_groups.get(group_name, [])
        if not gene_idx:
            continue
        sub_matrix = delta_tau_matrix[gene_idx, :]
        im = ax.imshow(
            sub_matrix,
            aspect='auto',
            cmap='RdBu_r',
            interpolation='nearest',
            extent=[pseudotime_grid.min(), pseudotime_grid.max(), 0, len(gene_idx)],
            origin='lower',
            vmin=-np.nanmax(np.abs(delta_tau_matrix)),
            vmax=np.nanmax(np.abs(delta_tau_matrix)),
        )
        ax.set_ylabel(f"{group_name}\n(n={len(gene_idx)})", fontsize=10)
    
    axes[-1].set_xlabel("Pseudotime", fontsize=12)
    fig.suptitle("Δτ heatmaps by gene group", fontsize=14)
    fig.colorbar(im, ax=axes, label="Δτ (pseudotime units)", shrink=0.8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig