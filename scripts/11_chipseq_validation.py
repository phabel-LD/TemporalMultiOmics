#!/usr/bin/env python
"""ChIP‑seq validation with faster CCF (fewer windows).

The transcription factor name is extracted from the target‑genes filename
(e.g. 'ASCL1_top500_genes.txt' → 'ASCL1') and used for the plot title
and output file name.
"""
import argparse, torch, numpy as np, pandas as pd
import scanpy as sc, anndata as ad
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import warnings
rng = np.random.default_rng(42) # Seed to eEnsure reproducible background set (the same across all runs)

from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface
from tmo.models import TMOLatentModelAsymmetric

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target_genes_file", required=True,
                        help="File with one gene symbol per line (e.g. ASCL1_top500_genes.txt)")
    parser.add_argument("--background_genes_file", default=None)
    parser.add_argument("--output_dir", default="./tmo_results")
    parser.add_argument("--n_components", type=int, default=50)
    parser.add_argument("--alternative", default="two-sided")
    parser.add_argument("--match_expression", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract TF name from target file name
    tf_name = Path(args.target_genes_file).stem.split("_")[0]   # "ASCL1_top500_genes" → "ASCL1"
    print(f"Running ChIP‑seq validation for {tf_name}")

    # -------- Load data --------
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    adata.var_names_make_unique()
    adata = adata[adata.obs['pseudotime'].argsort()].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA / LSI
    print("Computing PCA and LSI...")
    pca = PCA(n_components=args.n_components, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray())
    atac_mat = adata.obsm['ATAC_gene'].toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=args.n_components, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # CCF (fast mode)
    print("Computing CCF surface (fast mode)...")
    window_centers = np.linspace(0.2, 0.8, 5)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_lsi.T,
        rna_series=rna_pca.T,
        pseudotime=adata.obs['pseudotime'].values,
        window_centers=window_centers,
        window_half_width=0.15, max_lag=0.3, n_lags=31, bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)
    comp_lag_ccf = delta_tau_smooth[:, -1]
    if adata.uns.get('pseudotime_reversed', False):
        comp_lag_ccf = -comp_lag_ccf

    # Gene‑level lags
    loadings = pca.components_
    abs_load = np.abs(loadings)
    norm_load = abs_load / (abs_load.sum(axis=0, keepdims=True) + 1e-8)
    gene_lag_ccf = (norm_load.T @ comp_lag_ccf).flatten()
    gene_lag_ccf = pd.Series(gene_lag_ccf, index=adata.var_names)

    # Target genes
    with open(args.target_genes_file) as f:
        target_genes = [line.strip() for line in f if line.strip()]
    target_genes = [g for g in target_genes if g in adata.var_names]
    print(f"Target genes found: {len(target_genes)}")
    if len(target_genes) == 0:
        raise ValueError("No target genes found in dataset.")

    # Background genes (reproducible via independent RNG)
    all_genes = set(adata.var_names)
    bg_genes = list(all_genes - set(target_genes))

    if args.match_expression:
        print("Matching expression...")
        expr = np.array(adata.X.mean(axis=0)).flatten()
        expr_series = pd.Series(expr, index=adata.var_names)
        target_expr = expr_series[target_genes].values
        target_expr_median = np.median(target_expr)
        bg_genes = [g for g in bg_genes if abs(expr_series[g] - target_expr_median) < 0.5]
        print(f"Expression‑matched background genes: {len(bg_genes)}")
    else:
        print(f"Background genes available: {len(bg_genes)}")

    target_lags = [gene_lag_ccf[g] for g in target_genes]
    bg_lags = [gene_lag_ccf[g] for g in bg_genes]

    stat, p = mannwhitneyu(target_lags, bg_lags, alternative=args.alternative)
    print(f"\n=== CCF‑based Δτ validation for {tf_name} ===")
    print(f"Target Δτ mean: {np.mean(target_lags):.6f}, background Δτ mean: {np.mean(bg_lags):.6f}")
    print(f"Mann‑Whitney U: {stat}, p = {p:.4e}")

    # Plot with dynamic title and filename
    fig, ax = plt.subplots(figsize=(6,6))
    parts = ax.violinplot([target_lags, bg_lags], positions=[1, 2],
                          showmeans=True, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(['lightblue', 'lightcoral'][i])
        pc.set_alpha(0.7)
    # Add jittered points
    for i, vals in enumerate([target_lags, bg_lags]):
        x = np.random.normal(i+1, 0.04, size=len(vals))
        ax.scatter(x, vals, alpha=0.3, s=5, color='black')
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Target', 'Background'])
    ax.set_ylabel("Δτ")
    ax.set_title(f"ChIP‑seq validation ({tf_name}): p = {p:.3e}")
    plt.tight_layout()
    save_path = output_dir / f"chipseq_validation_{tf_name}_delta_tau.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved to {save_path}")

if __name__ == "__main__":
    main()