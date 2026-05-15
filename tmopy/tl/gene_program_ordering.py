"""Order components by Δτ at late pseudotime and perform GO enrichment."""

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import textwrap
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import gseapy as gp

from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface


def gene_program_ordering(
    adata: ad.AnnData,
    n_top_components: int = 10,
    organism: str = 'human',
    save_plots: str = None,
) -> pd.DataFrame:
    """
    Order components by Δτ at the last pseudotime window, separate most positive
    and most negative groups, and perform GO enrichment.

    Parameters
    ----------
    adata : AnnData
        Must contain .obsm['ATAC_gene'], .obs['pseudotime'].
    n_top_components : int, default=10
        Number of components to take from each end.
    organism : str, default='human'
        'human' or 'mouse' for GO enrichment.
    save_plots : str, optional
        Prefix to save the GO comparison bar plot.

    Returns
    -------
    pd.DataFrame
        Ranking of components by late Δτ, with top GO terms.
    """
    # Normalise RNA if needed
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
    late_lag = delta_tau_smooth[:, -1]

    # Load component annotation
    if 'component_annotation' not in adata.uns:
        raise ValueError("Component annotation missing. Run annotate_components first.")
    annot_df = adata.uns['component_annotation']

    # Sort components by late lag
    sorted_idx = np.argsort(late_lag)
    comp_order = np.arange(n_pca)[sorted_idx]
    late_lag_sorted = late_lag[sorted_idx]

    bottom_idx = comp_order[:n_top_components]   # most negative (RNA‑led)
    top_idx = comp_order[-n_top_components:]     # most positive (ATAC‑led)

    # Collect top genes
    def collect_genes(indices):
        genes = []
        for comp in indices:
            comp_name = f"Comp{comp}"
            row = annot_df[annot_df['Component'] == comp_name]
            if not row.empty:
                top_genes = row['Top_genes'].iloc[0].split(', ')
                genes.extend(top_genes)
        return list(set(genes))

    bottom_genes = collect_genes(bottom_idx)
    top_genes = collect_genes(top_idx)

    # GO enrichment
    print("Running GO enrichment for negative-lag components (RNA-led)...")
    enr_bottom = gp.enrichr(gene_list=bottom_genes,
                            gene_sets=['GO_Biological_Process_2023'],
                            organism=organism,
                            outdir=None,
                            cutoff=0.05)
    print("Running GO enrichment for positive-lag components (ATAC-led)...")
    enr_top = gp.enrichr(gene_list=top_genes,
                         gene_sets=['GO_Biological_Process_2023'],
                         organism=organism,
                         outdir=None,
                         cutoff=0.05)

    if enr_bottom.results is not None and not enr_bottom.results.empty:
        enr_bottom.results.to_csv("go_enrichment_negative_lag.csv", index=False)
    if enr_top.results is not None and not enr_top.results.empty:
        enr_top.results.to_csv("go_enrichment_positive_lag.csv", index=False)

    # Bar plot
    def get_top_terms(enr_res, n=5):
        if enr_res.results is None or enr_res.results.empty:
            return [], []
        res = enr_res.results.sort_values('Combined Score', ascending=False).head(n)
        return res['Term'].tolist(), res['Combined Score'].tolist()

    terms_bottom, scores_bottom = get_top_terms(enr_bottom, 5)
    terms_top, scores_top = get_top_terms(enr_top, 5)

    def wrap_labels(terms, width=40):
        return ['\n'.join(textwrap.wrap(t, width)) for t in terms]

    if save_plots:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        if terms_top:
            axes[0].barh(wrap_labels(terms_top)[::-1], scores_top[::-1], color='lightblue')
            axes[0].set_title('Positive Δτ components (ATAC‑led)')
        else:
            axes[0].text(0.5,0.5,'No enrichment', ha='center')
        if terms_bottom:
            axes[1].barh(wrap_labels(terms_bottom)[::-1], scores_bottom[::-1], color='lightcoral')
            axes[1].set_title('Negative Δτ components (RNA‑led)')
        else:
            axes[1].text(0.5,0.5,'No enrichment', ha='center')
        plt.tight_layout()
        plt.savefig(f"{save_plots}_gene_program_go_comparison.pdf", dpi=150)
        plt.close()

    # Ranking table
    comp_summary = []
    for i, comp in enumerate(comp_order):
        comp_name = f"Comp{comp}"
        row = annot_df[annot_df['Component'] == comp_name]
        go = row['GO_terms'].iloc[0].split(';')[0].strip() if not row.empty and pd.notna(row['GO_terms'].iloc[0]) else ""
        comp_summary.append({
            'Rank': i+1,
            'Component': comp_name,
            'Δτ (late)': f"{late_lag_sorted[i]:.3f}",
            'Top GO term': go
        })
    df = pd.DataFrame(comp_summary)
    return df