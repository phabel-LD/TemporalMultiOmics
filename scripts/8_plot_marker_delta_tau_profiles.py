#!/usr/bin/env python
"""Plot CCF-derived Δτ profiles for marker genes, with mean Δτ reference line."""

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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")

from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to processed AnnData")
    parser.add_argument("--markers", type=str, nargs='+', required=True,
                        help="List of marker gene names")
    parser.add_argument("--output_dir", type=str, default="./tmo_results",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------------------
    # 1. Load data, sort by pseudotime and preprocess
    # --------------------------------------------------------------------
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    if 'pseudotime' not in adata.obs:
        raise KeyError("'pseudotime' missing in .obs")
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_pca = 50
    pca = PCA(n_components=n_pca, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    # ATAC LSI
    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_pca, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # CCF Δτ surface
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
    # delta_tau_smooth shape: (n_components, n_windows) – each row is an RNA component

    # Mean Δτ across all components (for reference)
    mean_delta_tau = delta_tau_smooth.mean(axis=0)

    # --------------------------------------------------------------------
    # 2. Map marker genes to RNA components
    # --------------------------------------------------------------------
    loadings = pca.components_   # (n_components, n_genes)
    gene_to_comp = {}
    for i, gene in enumerate(adata.var_names):
        comp = np.argmax(np.abs(loadings[:, i]))
        gene_to_comp[gene] = comp

    # Load component annotations (for nicer labels)
    annot_file = output_dir / "component_annotation.csv"
    if annot_file.exists():
        annot_df = pd.read_csv(annot_file)
    else:
        annot_df = None
        print("Annotation file not found; using generic component labels.")

    # --------------------------------------------------------------------
    # 3. Plot profiles
    # --------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(args.markers), figsize=(5*len(args.markers), 4), sharey=True)
    if len(args.markers) == 1:
        axes = [axes]

    for ax, gene in zip(axes, args.markers):
        if gene not in gene_to_comp:
            print(f"Warning: {gene} not found. Skipping.")
            continue
        comp = gene_to_comp[gene]
        # Get component label
        if annot_df is not None:
            row = annot_df[annot_df['Component'] == f"Comp{comp}"]
            if not row.empty:
                go_term = row['GO_terms'].iloc[0].split(';')[0].strip() if pd.notna(row['GO_terms'].iloc[0]) else ""
                label = f"Comp{comp} ({go_term})" if go_term else f"Comp{comp}"
            else:
                label = f"Comp{comp}"
        else:
            label = f"Comp{comp}"
        # Profile
        profile = delta_tau_smooth[comp, :]
        ax.plot(window_centers, profile, 'o-', color='steelblue', linewidth=2, markersize=6, label=f"{gene} (Δτ)")
        ax.plot(window_centers, mean_delta_tau, 'k--', alpha=0.5, linewidth=1.5, label='Mean of all comps')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel("Pseudotime")
        ax.set_ylabel("Δτ (regulatory lag)")
        ax.set_title(f"{gene}\n{label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = output_dir / "marker_delta_tau_profiles.pdf"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"Saved marker Δτ profiles with mean line to {out_file}")

if __name__ == "__main__":
    main()