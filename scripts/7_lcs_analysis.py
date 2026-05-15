#!/usr/bin/env python
"""LCS analysis: scatter plots and bar chart comparing asymmetric vs symmetric model.

Uses the same model class for both (TMOLatentModelAsymmetric) and reads
the best validation LCS from the training metrics CSV files.
"""

import argparse
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface


def compute_lcs(pred_lags, target_lags):
    valid = ~np.isnan(target_lags)
    if valid.sum() < 2:
        return np.nan
    return spearmanr(pred_lags[valid], target_lags[valid])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--asym_model", type=str, default="./tmo_results/tmo_asymmetric_best.pt")
    parser.add_argument("--sym_model", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cpu")

    # -------------------------------------------------------------
    # 1. Load data and compute latent representations (full dataset)
    # -------------------------------------------------------------
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
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
    n_atac = n_pca
    lsi = TruncatedSVD(n_components=n_atac, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # CCF surface
    atac_comp_T = atac_lsi.T
    rna_comp_T = rna_pca.T
    pt = adata.obs['pseudotime'].values
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_comp_T, rna_series=rna_comp_T,
        pseudotime=pt, window_centers=window_centers,
        window_half_width=0.1, max_lag=0.3, n_lags=31, bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)
    ccf_lag = delta_tau_smooth[:, -1]   # last window

    # Handle pseudotime reversal
    if adata.uns.get('pseudotime_reversed', False):
        ccf_lag = -ccf_lag

    # Tensors
    rna_t = torch.tensor(rna_pca, dtype=torch.float32)
    atac_t = torch.tensor(atac_lsi, dtype=torch.float32)
    pt_t = torch.tensor(pt, dtype=torch.float32)

    # -------------------------------------------------------------
    # 2. Helper to get model LCS and predictions
    # -------------------------------------------------------------
    def evaluate_model(model_path, use_bias):
        model = TMOLatentModelAsymmetric(
            n_rna_components=n_pca, n_atac_components=n_atac,
            d_model=64, n_heads=4, num_layers=2, dropout=0.1,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            out = model(rna_t.to(device), atac_t.to(device), pt_t.to(device), use_bias=use_bias)
            pred = out['lag_per_token'][:, :n_atac].cpu().numpy().mean(axis=0)
        lcs = compute_lcs(pred, ccf_lag)
        return lcs, pred

    # -------------------------------------------------------------
    # 3. Asymmetric model
    # -------------------------------------------------------------
    print("Evaluating asymmetric model...")
    lcs_asym, pred_asym = evaluate_model(args.asym_model, use_bias=True)
    print(f"Asymmetric LCS (full data) = {lcs_asym:.4f}")

    # -------------------------------------------------------------
    # 4. Symmetric model (if provided)
    # -------------------------------------------------------------
    lcs_sym = None
    pred_sym = None
    if args.sym_model:
        print("Evaluating symmetric model...")
        lcs_sym, pred_sym = evaluate_model(args.sym_model, use_bias=False)
        print(f"Symmetric LCS (full data) = {lcs_sym:.4f}")

    # -------------------------------------------------------------
    # 5. Scatter plots
    # -------------------------------------------------------------
    # Asymmetric scatter
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(ccf_lag, pred_asym, alpha=0.7)
    m, b = np.polyfit(ccf_lag, pred_asym, 1)
    ax.plot(ccf_lag, m*ccf_lag + b, 'r--', label=f'slope={m:.3f}')
    ax.set_xlabel("CCF target Δτ")
    ax.set_ylabel("Predicted Δτ")
    ax.set_title(f"Asymmetric model (LCS = {lcs_asym:.3f})")
    ax.legend()
    plt.savefig(output_dir / "lcs_scatter_asymmetric.png", dpi=150)
    plt.close()

    # Symmetric scatter (if available)
    if pred_sym is not None:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(ccf_lag, pred_sym, alpha=0.7, color='orange')
        m, b = np.polyfit(ccf_lag, pred_sym, 1)
        ax.plot(ccf_lag, m*ccf_lag + b, 'r--', label=f'slope={m:.3f}')
        ax.set_xlabel("CCF target Δτ")
        ax.set_ylabel("Predicted Δτ")
        ax.set_title(f"Symmetric model (LCS = {lcs_sym:.3f})")
        ax.legend()
        plt.savefig(output_dir / "lcs_scatter_symmetric.png", dpi=150)
        plt.close()

    # -------------------------------------------------------------
    # 6. Bar chart using the best validation LCS from training CSVs
    # -------------------------------------------------------------
    asym_csv = output_dir / "training_metrics_tmo_asymmetric.csv"
    sym_csv = output_dir / "training_metrics_tmo_symmetric.csv"

    best_asym = np.nan
    if asym_csv.exists():
        df = pd.read_csv(asym_csv)
        if 'lcs' in df.columns:
            best_asym = df['lcs'].max()
    best_sym = np.nan
    if sym_csv and sym_csv.exists():
        df = pd.read_csv(sym_csv)
        if 'lcs' in df.columns:
            best_sym = df['lcs'].max()

    models = ['Asymmetric']
    values = [best_asym]
    if not np.isnan(best_sym):
        models.append('Symmetric')
        values.append(best_sym)

    fig, ax = plt.subplots(figsize=(4,6))
    bars = ax.bar(models, values, color=['steelblue', 'darkorange'][:len(models)])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Best validation LCS")
    ax.set_title("Lag Concordance Score")
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha='center', va='bottom')
    plt.savefig(output_dir / "lcs_comparison.png", dpi=150)
    plt.close()

    print("LCS analysis finished. Figures saved in", output_dir)


if __name__ == "__main__":
    main()