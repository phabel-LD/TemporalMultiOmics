"""Compare LCS between asymmetric and symmetric models (scatter plots, bar chart).

This module provides a function that loads trained asymmetric and symmetric
TMO models, computes the Lag Concordance Score (LCS) on the full dataset,
and generates diagnostic plots that illustrate the quality of the learned
regulatory lags.

The LCS is the Spearman correlation between the per‑component mean
predicted lags and the CCF‑derived target lags.  A high positive LCS
indicates that the model correctly orders the components by their
regulatory timing.

Plots produced (saved to disk when ``save_plots`` is provided):
    - Scatter plot of predicted vs CCF lags for the asymmetric model,
      with a linear regression line.
    - Scatter plot for the symmetric model (if provided).
    - Bar chart comparing the LCS values of the two models.

Important note:
    The LCS reported here is computed on the full dataset and is meant
    for qualitative visual inspection.  The official LCS values used in
    the paper are recorded in the training metrics CSV files and represent
    the model’s performance on the same data it was trained on (in‑set
    diagnostic).  Independent biological validation is provided by the
    ChIP‑seq and Perturb‑seq experiments.
"""

import torch
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from pathlib import Path

from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface


def lcs_analysis(
    adata: ad.AnnData,
    asym_model_path: str,
    sym_model_path: str = None,
    n_components: int = 50,
    save_plots: str = None,
):
    """Compute LCS for asymmetric (and optionally symmetric) models and generate plots.

    The function performs the following steps:
        1. Normalise RNA and compute PCA/LSI projections on the full dataset.
        2. Compute the CCF surface and extract the target lags.
        3. Load the trained model(s) and predict per‑component mean lags.
        4. Compute the Spearman correlation between predicted and target lags.
        5. Optionally create scatter plots and a bar chart.

    Parameters
    ----------
    adata : AnnData
        Must contain ``.obs['pseudotime']`` and ``.obsm['ATAC_gene']``.
    asym_model_path : str
        Path to the trained asymmetric model checkpoint (``.pt``).
    sym_model_path : str, optional
        Path to the trained symmetric baseline model checkpoint.  If not
        provided, only the asymmetric model is evaluated.
    n_components : int, default=50
        Number of PCA/LSI components.
    save_plots : str, optional
        If provided, save the generated plots using this string as a
        prefix.  The files ``<save_plots>_lcs_scatter_asymmetric.png``,
        ``<save_plots>_lcs_scatter_symmetric.png``, and
        ``<save_plots>_lcs_comparison.png`` are created.

    Returns
    -------
    dict
        ``{'lcs_asym': float, 'lcs_sym': float or None}``
    """

    # Ensure RNA is log‑normalised
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ----------------------------------------------------------------
    # Latent projections (mirror those used during training)
    # ----------------------------------------------------------------
    pca = PCA(n_components=n_components, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    atac_mat = adata.obsm['ATAC_gene'].toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # ----------------------------------------------------------------
    # CCF surface -> target lags (last pseudotime window)
    # ----------------------------------------------------------------
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
    ccf_lag = delta_tau_smooth[:, -1] # target: last window

    device = torch.device("cpu")

    # ----------------------------------------------------------------
    # Helper to load a model and obtain per‑component mean lags
    # ----------------------------------------------------------------
    def get_model_lcs(model_path, use_bias):
        model = TMOLatentModelAsymmetric(
            n_rna_components=n_components,
            n_atac_components=n_components,
            d_model=64, n_heads=4, num_layers=2, dropout=0.1,
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(rna_pca, dtype=torch.float32),
                        torch.tensor(atac_lsi, dtype=torch.float32),
                        torch.tensor(pt, dtype=torch.float32),
                        use_bias=use_bias)
            pred_lags = out['lag_per_token'][:, :n_components].numpy().mean(axis=0)
        valid = ~np.isnan(ccf_lag)
        lcs = spearmanr(pred_lags[valid], ccf_lag[valid])[0]
        return lcs, pred_lags

    # Evaluate asymmetric model
    lcs_asym, pred_asym = get_model_lcs(asym_model_path, use_bias=True)
    print(f"Asymmetric LCS (full data) = {lcs_asym:.4f}")

    # Evaluate symmetric model (if provided)
    lcs_sym = None
    pred_sym = None
    if sym_model_path:
        lcs_sym, pred_sym = get_model_lcs(sym_model_path, use_bias=False)
        print(f"Symmetric LCS (full data) = {lcs_sym:.4f}")

    # ----------------------------------------------------------------
    # Generate diagnostic plots
    # ----------------------------------------------------------------
    if save_plots:
        # --- Asymmetric scatter ---
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(ccf_lag, pred_asym, alpha=0.7)
        m, b = np.polyfit(ccf_lag, pred_asym, 1)
        ax.plot(ccf_lag, m*ccf_lag + b, 'r--', label=f'slope={m:.3f}')
        ax.set_xlabel("CCF lag")
        ax.set_ylabel("Predicted lag")
        ax.set_title(f"Asymmetric: LCS = {lcs_asym:.3f}")
        ax.legend()
        plt.savefig(f"{save_plots}_lcs_scatter_asymmetric.png", dpi=150)
        plt.close()

        # --- Symmetric scatter (if available) ---
        if pred_sym is not None:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(ccf_lag, pred_sym, alpha=0.7, color='orange')
            m, b = np.polyfit(ccf_lag, pred_sym, 1)
            ax.plot(ccf_lag, m*ccf_lag + b, 'r--', label=f'slope={m:.3f}')
            ax.set_xlabel("CCF lag")
            ax.set_ylabel("Predicted lag")
            ax.set_title(f"Symmetric: LCS = {lcs_sym:.3f}")
            ax.legend()
            plt.savefig(f"{save_plots}_lcs_scatter_symmetric.png", dpi=150)
            plt.close()

        # --- Bar chart ---
        models = ['Asymmetric']
        values = [lcs_asym]
        if lcs_sym is not None:
            models.append('Symmetric')
            values.append(lcs_sym)
        fig, ax = plt.subplots(figsize=(4,6))
        ax.bar(models, values, color=['blue', 'red'][:len(models)])
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("LCS (full data)")
        ax.set_title("Lag Concordance Score")
        for i, v in enumerate(values):
            ax.text(i, v+0.02, f"{v:.3f}", ha='center')
        plt.savefig(f"{save_plots}_lcs_comparison.png", dpi=150)
        plt.close()

    return {'lcs_asym': lcs_asym, 'lcs_sym': lcs_sym}