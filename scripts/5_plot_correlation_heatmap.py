#!/usr/bin/env python
"""Generate Figure 2: maximum correlation between RNA components and any ATAC component over pseudotime bins."""

import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import pearsonr
from pathlib import Path
import pandas as pd   # added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of pseudotime bins")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    if 'pseudotime' not in adata.obs:
        raise KeyError("'pseudotime' missing in .obs")
    adata = adata[adata.obs['pseudotime'].argsort()].copy()

    # Normalise RNA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # RNA PCA
    n_rna_pca = 50
    pca = PCA(n_components=n_rna_pca, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    # ATAC LSI
    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    n_atac_lsi = 50
    lsi = TruncatedSVD(n_components=n_atac_lsi, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # Bin cells by pseudotime
    pt = adata.obs['pseudotime'].values
    bins = np.linspace(0, 1, args.n_bins + 1)
    bin_indices = np.digitize(pt, bins) - 1

    # For each bin, for each RNA component, compute max absolute correlation with any ATAC component
    max_corr_per_bin = []  # (n_bins, n_rna_pca)
    for b in range(args.n_bins):
        mask = (bin_indices == b)
        if mask.sum() < 3:
            max_corr_per_bin.append(np.zeros(n_rna_pca))
            continue
        atac_b = atac_lsi[mask]
        rna_b = rna_pca[mask]
        # Correlation matrix: (n_atac, n_rna)
        corr_mat = np.corrcoef(atac_b.T, rna_b.T)[:n_atac_lsi, n_atac_lsi:]
        # For each RNA component, take maximum absolute correlation over ATAC components
        max_corr = np.max(np.abs(corr_mat), axis=0)  # (n_rna_pca,)
        max_corr_per_bin.append(max_corr)
    max_corr_per_bin = np.array(max_corr_per_bin)  # (n_bins, n_rna)

    # Save the correlation matrix for downstream analysis
    np.savetxt(str(output_dir / "atac_rna_correlation_heatmap.csv"), max_corr_per_bin, delimiter=",")
    print(f"Correlation matrix saved to {output_dir / 'atac_rna_correlation_heatmap.csv'}")

    # ===== Load component annotations =====
    annot_file = output_dir / "component_annotation.csv"
    if annot_file.exists():
        annot_df = pd.read_csv(annot_file)
        # Create mapping from Component (e.g., "Comp0") to short label
        comp_to_label = {}
        for _, row in annot_df.iterrows():
            comp = row['Component']
            go_term = row['GO_terms'].split(';')[0].strip() if pd.notna(row['GO_terms']) else ""
            if go_term:
                label = f"{comp} ({go_term[:30]})"
            else:
                label = comp
            comp_to_label[comp] = label
        # Generate labels for all 50 components (some may not be in CSV; use generic)
        labels = [comp_to_label.get(f"Comp{i}", f"Comp{i}") for i in range(n_rna_pca)]
    else:
        print("Annotation file not found; using generic labels.")
        labels = [f"Comp{i}" for i in range(n_rna_pca)]
    # ====================================

    # Plot heatmap (RNA components vs pseudotime bins)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(max_corr_per_bin.T, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=0.8)
    ax.set_xlabel("Pseudotime bin")
    ax.set_ylabel("RNA component")
    ax.set_title("Maximum absolute correlation between RNA component and any ATAC component")
    ax.set_xticks(range(args.n_bins))
    ax.set_xticklabels([f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(args.n_bins)], rotation=45)
    # Use annotated labels on y-axis
    ax.set_yticks(range(n_rna_pca))
    ax.set_yticklabels(labels, fontsize=8)
    plt.colorbar(im, label="Max |correlation|")
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_correlation_max_over_pseudotime.pdf")
    plt.close()
    print(f"Figure 2 saved to {output_dir / 'fig2_correlation_max_over_pseudotime.pdf'}")

if __name__ == "__main__":
    main()