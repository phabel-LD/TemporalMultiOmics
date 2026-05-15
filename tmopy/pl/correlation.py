"""ATAC‑RNA correlation heatmap (Figure 2)."""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD


def plot_correlation(
    adata: ad.AnnData,
    n_bins: int = 10,
    save: str = None,
    show: bool = True,
    figsize: tuple = (12, 8),
    vmax: float = 0.8,
):
    """
    Generate ATAC‑RNA correlation heatmap (U‑shape).

    Parameters
    ----------
    adata : AnnData
        Must contain .obsm['ATAC_gene'] and .obs['pseudotime'].
    n_bins : int, default=10
        Number of pseudotime bins.
    save : str, optional
        Path to save the figure (e.g., 'fig2_correlation.pdf').
    show : bool, default=True
        Whether to display the figure.
    figsize : tuple, default=(12,8)
        Figure size.
    vmax : float, default=0.8
        Maximum value for colour scale (minimum fixed at 0).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    """
    # Ensure RNA is log‑normalised
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # RNA PCA (50 components)
    n_rna_pca = 50
    pca = PCA(n_components=n_rna_pca, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    # ATAC LSI (50 components) from gene‑level ATAC matrix
    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    n_atac_lsi = 50
    lsi = TruncatedSVD(n_components=n_atac_lsi, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # Bin pseudotime
    pt = adata.obs['pseudotime'].values
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(pt, bins) - 1

    # For each bin, compute max absolute correlation between RNA and any ATAC component
    max_corr_per_bin = []   # shape (n_bins, n_rna_pca)
    for b in range(n_bins):
        mask = (bin_indices == b)
        if mask.sum() < 3:
            max_corr_per_bin.append(np.zeros(n_rna_pca))
            continue
        atac_b = atac_lsi[mask]
        rna_b = rna_pca[mask]
        # Correlation matrix (n_atac, n_rna)
        corr_mat = np.corrcoef(atac_b.T, rna_b.T)[:n_atac_lsi, n_atac_lsi:]
        max_corr = np.max(np.abs(corr_mat), axis=0)   # (n_rna_pca,)
        max_corr_per_bin.append(max_corr)
    max_corr_per_bin = np.array(max_corr_per_bin)   # (n_bins, n_rna)

    # Row labels (RNA components) from component annotation if available
    if 'component_annotation' in adata.uns:
        annot_df = adata.uns['component_annotation']
        comp_to_label = {}
        for _, row in annot_df.iterrows():
            comp = row['Component']
            go_term = row['GO_terms'].split(';')[0].strip() if pd.notna(row['GO_terms']) else ""
            if go_term:
                comp_to_label[comp] = f"{comp} ({go_term[:30]})"
            else:
                comp_to_label[comp] = comp
        labels = [comp_to_label.get(f"Comp{i}", f"Comp{i}") for i in range(n_rna_pca)]
    else:
        labels = [f"Comp{i}" for i in range(n_rna_pca)]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(max_corr_per_bin.T, aspect='auto', cmap='viridis', origin='lower',
                   vmin=0, vmax=vmax, interpolation='nearest')
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("RNA component")
    ax.set_title("Maximum absolute correlation between RNA component and any ATAC component")
    ax.set_xticks(range(n_bins))
    ax.set_xticklabels([f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)], rotation=45)
    ax.set_yticks(range(n_rna_pca))
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, label="Max |correlation|")
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig