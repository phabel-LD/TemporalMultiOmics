"""Plot Δτ profiles for a list of marker genes (from CCF)."""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface


def plot_marker_profiles(
    adata: ad.AnnData,
    marker_genes: list,
    save: str = None,
    show: bool = True,
    figsize_per_gene: float = 5,
):
    """
    Plot Δτ profiles for marker genes (with mean Δτ reference line).

    Parameters
    ----------
    adata : AnnData
        Must contain .obsm['ATAC_gene'], .obs['pseudotime'].
    marker_genes : list of str
        Gene symbols (must be in adata.var_names).
    save : str, optional
        Path to save the figure.
    show : bool, default=True
        Whether to display the figure.
    figsize_per_gene : float, default=5
        Width per subplot.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Normalise RNA
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    n_pca = 50
    pca = PCA(n_components=n_pca, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    atac_mat = adata.obsm['ATAC_gene'].toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_pca, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # CCF surface
    atac_comp_T = atac_lsi.T
    rna_comp_T = rna_pca.T
    pt = adata.obs['pseudotime'].values
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_comp_T,
        rna_series=rna_comp_T,
        pseudotime=pt,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)
    if adata.uns.get('pseudotime_reversed', False):
        delta_tau_smooth = -delta_tau_smooth
    mean_delta_tau = delta_tau_smooth.mean(axis=0)

    # Map marker genes to components
    loadings = pca.components_
    gene_to_comp = {}
    for i, gene in enumerate(adata.var_names):
        comp = np.argmax(np.abs(loadings[:, i]))
        gene_to_comp[gene] = comp

    # Load component annotation for labels
    if 'component_annotation' in adata.uns:
        annot_df = adata.uns['component_annotation']
        comp_to_label = {}
        for _, row in annot_df.iterrows():
            comp = row['Component']
            go_term = row['GO_terms'].split(';')[0].strip() if pd.notna(row['GO_terms']) else ""
            comp_to_label[comp] = go_term

    # Plot
    n_markers = len(marker_genes)
    fig, axes = plt.subplots(n_markers, 1, figsize=(8, figsize_per_gene * n_markers), sharex=True, sharey=False)
    if n_markers == 1:
        axes = [axes]

    for ax, gene in zip(axes, marker_genes):
        if gene not in gene_to_comp:
            print(f"Warning: {gene} not found, skipping.")
            continue
        comp = gene_to_comp[gene]
        label = comp_to_label.get(f"Comp{comp}", f"Comp{comp}") if 'comp_to_label' in locals() else f"Comp{comp}"
        profile = delta_tau_smooth[comp, :]
        ax.plot(window_centers, profile, 'o-', color='steelblue', label=f"{gene}")
        ax.plot(window_centers, mean_delta_tau, 'k--', alpha=0.5, label='Mean Δτ')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Δτ (regulatory lag)")
        ax.set_title(f"{gene}\n{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    return fig