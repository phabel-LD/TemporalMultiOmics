#!/usr/bin/env python
"""Order components by Δτ at late pseudotime and perform GO enrichment."""

import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import gseapy as gp
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")

from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--n_top_components", type=int, default=10)
    parser.add_argument("--organism", type=str, default='human',
                        help="Species for GO enrichment: 'human', 'mouse', etc.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # 1. Load data and sort by pseudotime
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    if 'pseudotime' not in adata.obs:
        raise KeyError("'pseudotime' missing in .obs")
    adata = adata[adata.obs['pseudotime'].argsort()].copy()   # sort by pseudotime
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_pca = 50
    pca = PCA(n_components=n_pca, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_pca, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # 2. Compute CCF (using sorted pseudotime)
    print("Computing CCF surface...")
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
    if 'pseudotime_reversed' in adata.uns and adata.uns['pseudotime_reversed']:
        delta_tau_smooth = -delta_tau_smooth
        print("Flipped sign of Δτ for reversed dataset.")
    late_lag = delta_tau_smooth[:, -1]

    # 3. Load component annotation
    annot_file = output_dir / "component_annotation.csv"
    if not annot_file.exists():
        raise FileNotFoundError("Run annotate_components.py first.")
    annot_df = pd.read_csv(annot_file)

    # 4. Order components by late lag (most negative first)
    sorted_idx = np.argsort(late_lag)
    comp_order = np.arange(n_pca)[sorted_idx]
    late_lag_sorted = late_lag[sorted_idx]

    bottom_idx = comp_order[:args.n_top_components]   # most negative
    top_idx = comp_order[-args.n_top_components:]    # most positive

    # Collect genes
    bottom_genes = []
    top_genes = []
    for comp in bottom_idx:
        comp_name = f"Comp{comp}"
        row = annot_df[annot_df['Component'] == comp_name]
        if not row.empty:
            genes = row['Top_genes'].iloc[0].split(', ')
            bottom_genes.extend(genes)
    for comp in top_idx:
        comp_name = f"Comp{comp}"
        row = annot_df[annot_df['Component'] == comp_name]
        if not row.empty:
            genes = row['Top_genes'].iloc[0].split(', ')
            top_genes.extend(genes)
    bottom_genes = list(set(bottom_genes))
    top_genes = list(set(top_genes))

    # 5. GO enrichment
    print("Running GO enrichment for negative-lag components (RNA-led)...")
    enr_bottom = gp.enrichr(gene_list=bottom_genes,
                            gene_sets=['GO_Biological_Process_2023'],
                            organism=args.organism,
                            outdir=None,
                            cutoff=0.05)
    print("Running GO enrichment for positive-lag components (ATAC-led)...")
    enr_top = gp.enrichr(gene_list=top_genes,
                         gene_sets=['GO_Biological_Process_2023'],
                         organism=args.organism,
                         outdir=None,
                         cutoff=0.05)

    if enr_bottom.results is not None and not enr_bottom.results.empty:
        enr_bottom.results.to_csv(output_dir / "go_enrichment_negative_lag.csv", index=False)
    if enr_top.results is not None and not enr_top.results.empty:
        enr_top.results.to_csv(output_dir / "go_enrichment_positive_lag.csv", index=False)

    # 6. Plot top 5 GO terms (vertical stacking) with adjusted margins and font size
    import textwrap

    def get_top_terms(enr_result, n=5):
        if enr_result.results is None or enr_result.results.empty:
            return [], []
        res = enr_result.results.sort_values('Combined Score', ascending=False).head(n)
        return res['Term'].tolist(), res['Combined Score'].tolist()

    terms_bottom, scores_bottom = get_top_terms(enr_bottom, 5)
    terms_top, scores_top = get_top_terms(enr_top, 5)

    # Wrap long GO term names to 40 characters (optional)
    def wrap_labels(terms, width=40):
        return ['\n'.join(textwrap.wrap(t, width)) for t in terms]

    if terms_top:
        terms_top_wrapped = wrap_labels(terms_top)
        terms_bottom_wrapped = wrap_labels(terms_bottom)
    else:
        terms_top_wrapped = []
        terms_bottom_wrapped = []

    fig, axes = plt.subplots(2, 1, figsize=(7, 7))   # slightly narrower figure
    # Top subplot: positive lag (ATAC‑led)
    if terms_top:
        bars = axes[0].barh(terms_top_wrapped[::-1], scores_top[::-1], color='lightblue')
        axes[0].set_xlabel('Combined Score', fontsize=9)
        axes[0].set_title('Positive Δτ components (ATAC‑led)', fontsize=10)
        axes[0].tick_params(axis='y', labelsize=7)
    else:
        axes[0].text(0.5, 0.5, 'No enrichment', ha='center')
    # Bottom subplot: negative lag (RNA‑led)
    if terms_bottom:
        axes[1].barh(terms_bottom_wrapped[::-1], scores_bottom[::-1], color='lightcoral')
        axes[1].set_xlabel('Combined Score', fontsize=9)
        axes[1].set_title('Negative Δτ components (RNA‑led)', fontsize=10)
        axes[1].tick_params(axis='y', labelsize=7)
    else:
        axes[1].text(0.5, 0.5, 'No enrichment', ha='center')
    
    # Adjust layout: increase left margin to 0.4, reduce right margin
    plt.subplots_adjust(left=0.4, right=0.95, top=0.9, bottom=0.1, hspace=0.5)
    plt.savefig(output_dir / "gene_program_go_comparison.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved GO comparison plot to {output_dir / 'gene_program_go_comparison.pdf'}")

    # 7. Ranking table
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
    pd.DataFrame(comp_summary).to_csv(output_dir / "component_ordering_by_late_lag.csv", index=False)
    print(f"Saved component ranking to {output_dir / 'component_ordering_by_late_lag.csv'}")

if __name__ == "__main__":
    main()