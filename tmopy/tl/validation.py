"""Causal validation using Perturb‑seq (δΔτ) and ChIP‑seq (CCF‑based Δτ).

Now uses pre‑fitted PCA/TF‑IDF/LSI from training to ensure identical latent spaces.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import mannwhitneyu
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process.kernels")

from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface


def _load_or_fit_transformers(adata, pca, tfidf, lsi, n_components):
    """Load pre‑fitted transformers or (if not provided) fit on the given data.

    Returns the transformers and the projected data.
    """
    # --- PCA ---
    if isinstance(pca, (str, Path)):
        with open(pca, 'rb') as f:
            pca = pickle.load(f)
    if pca is None:
        warnings.warn("PCA not provided; fitting on current data – may not match training space.")
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    rna_pca = pca.transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    # --- TF‑IDF ---
    if isinstance(tfidf, (str, Path)):
        with open(tfidf, 'rb') as f:
            tfidf = pickle.load(f)
    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    if tfidf is None:
        warnings.warn("TF‑IDF not provided; fitting on current data – may not match training space.")
        tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
        tfidf.fit(atac_mat)
    atac_tfidf = tfidf.transform(atac_mat)

    # --- LSI ---
    if isinstance(lsi, (str, Path)):
        with open(lsi, 'rb') as f:
            lsi = pickle.load(f)
    if lsi is None:
        warnings.warn("LSI not provided; fitting on current data – may not match training space.")
        lsi = TruncatedSVD(n_components=n_components, random_state=42)
        lsi.fit(atac_tfidf)
    atac_lsi = lsi.transform(atac_tfidf)

    return pca, tfidf, lsi, rna_pca, atac_lsi


def validate_perturbseq(
    adata,
    model_path,
    control_label,
    perturb_label,
    target_genes,
    background_genes=None,
    match_expression=True,
    n_components=50,
    perturbation_column='guide_target',
    save_plot=None,
    pca=None,           # path or pre‑fitted PCA
    tfidf=None,         # path or pre‑fitted TfidfTransformer
    lsi=None,           # path or pre‑fitted TruncatedSVD
):
    """Causal validation for a Perturb‑seq experiment (δΔτ multi‑gene test).

    Now uses pre‑fitted PCA/TF‑IDF/LSI from training to guarantee consistent
    latent spaces. If not provided, a warning is emitted and they are fitted
    on the current (possibly perturbed) data.
    """
    # Normalise if needed
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Split cells
    control = adata[adata.obs[perturbation_column] == control_label]
    perturbed = adata[adata.obs[perturbation_column] == perturb_label]
    print(f"Control cells: {control.n_obs}, Perturbed cells: {perturbed.n_obs}")

    # Load or fit transformers (applied to all cells, then split by index)
    pca, tfidf, lsi, rna_pca, atac_lsi = _load_or_fit_transformers(
        adata, pca, tfidf, lsi, n_components
    )

    # Indices for control and perturbed
    ctrl_idx = [adata.obs_names.get_loc(b) for b in control.obs_names]
    pert_idx = [adata.obs_names.get_loc(b) for b in perturbed.obs_names]

    rna_ctrl = rna_pca[ctrl_idx]
    rna_pert = rna_pca[pert_idx]
    atac_ctrl = atac_lsi[ctrl_idx]
    atac_pert = atac_lsi[pert_idx]
    pt_ctrl = control.obs['pseudotime'].values
    pt_pert = perturbed.obs['pseudotime'].values

    # Load model
    device = torch.device("cpu")
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_components,
        n_atac_components=n_components,
        d_model=64, n_heads=4, num_layers=2, dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Predict lags
    with torch.no_grad():
        out_ctrl = model(torch.tensor(rna_ctrl, dtype=torch.float32),
                         torch.tensor(atac_ctrl, dtype=torch.float32),
                         torch.tensor(pt_ctrl, dtype=torch.float32),
                         use_bias=True)
        out_pert = model(torch.tensor(rna_pert, dtype=torch.float32),
                         torch.tensor(atac_pert, dtype=torch.float32),
                         torch.tensor(pt_pert, dtype=torch.float32),
                         use_bias=True)
        lag_ctrl = out_ctrl['lag_per_token'][:, :n_components].numpy().mean(axis=0)
        lag_pert = out_pert['lag_per_token'][:, :n_components].numpy().mean(axis=0)

    delta_tau_shift = np.abs(lag_pert - lag_ctrl)

    # Map target genes to components using PCA loadings
    loadings = pca.components_
    gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}
    target_comps = []
    for g in target_genes:
        if g not in gene_to_idx:
            print(f"Warning: {g} not found, skipping.")
            continue
        idx = gene_to_idx[g]
        comp_weights = np.abs(loadings[:, idx])
        best_comp = np.argmax(comp_weights)
        target_comps.append(best_comp)
        print(f"{g} -> component {best_comp} (loading {comp_weights[best_comp]:.4f})")
    if not target_comps:
        raise ValueError("No target genes found in dataset.")
    target_comps = list(set(target_comps))

    # Background components
    if background_genes is None:
        background_comps = [i for i in range(n_components) if i not in target_comps]
    else:
        background_comps = []
        for g in background_genes:
            if g not in gene_to_idx: continue
            idx = gene_to_idx[g]
            comp_weights = np.abs(loadings[:, idx])
            best_comp = np.argmax(comp_weights)
            background_comps.append(best_comp)
        background_comps = list(set(background_comps))

    # Expression matching (optional)
    if match_expression:
        expr = adata.X.mean(axis=0)
        if hasattr(expr, 'A1'):
            expr = expr.A1
        else:
            expr = np.asarray(expr).flatten()
        expr_series = pd.Series(expr, index=adata.var_names)
        target_expr = [expr_series[g] for g in target_genes if g in expr_series]
        if target_expr:
            target_median = np.median(target_expr)
            possible_bg = [g for g in adata.var_names if g not in target_genes
                           and abs(expr_series[g] - target_median) < 0.5]
            if possible_bg:
                background_comps = []
                for g in possible_bg:
                    idx = gene_to_idx[g]
                    comp_weights = np.abs(loadings[:, idx])
                    best_comp = np.argmax(comp_weights)
                    background_comps.append(best_comp)
                background_comps = list(set(background_comps))

    target_shifts = delta_tau_shift[target_comps]
    background_shifts = delta_tau_shift[background_comps]

    stat, p = mannwhitneyu(target_shifts, background_shifts, alternative='greater')
    print(f"Target δΔτ mean: {target_shifts.mean():.6f}, background mean: {background_shifts.mean():.6f}")
    print(f"p (target > background) = {p:.4e}")

    if save_plot:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.violinplot([target_shifts, background_shifts], positions=[1,2],
                      showmeans=True, showmedians=True, widths=0.7)
        ax.set_xticks([1,2])
        ax.set_xticklabels(['Target','Background'])
        ax.set_ylabel("δΔτ (pseudotime units)")
        ax.set_title(f"Perturbation validation (p = {p:.3e})")
        plt.savefig(save_plot, dpi=150)
        plt.close()

    return {
        'p_value': p,
        'target_shifts': target_shifts,
        'background_shifts': background_shifts,
        'mean_target': target_shifts.mean(),
        'mean_background': background_shifts.mean(),
    }


def validate_chipseq(
    adata,
    model_path,
    target_genes_file,
    background_genes_file=None,
    match_expression=False,
    n_components=50,
    alternative='two-sided',
    save_plot=None,
    pca=None,
    tfidf=None,
    lsi=None,
):
    """
    ChIP‑seq validation using a pre‑computed target gene list (deterministic).

    Compares the CCF‑derived gene‑level Δτ of the target genes against the
    **full set of eligible background genes** (all non‑target genes, optionally
    expression‑matched). No random subsampling is performed.

    Parameters
    ----------
    adata : AnnData
        Must contain .obs['pseudotime'] and .obsm['ATAC_gene'].
    model_path : str
        Path to trained asymmetric model (.pt).
    target_genes_file : str
        Path to a text file with one gene symbol per line.
    background_genes_file : str, optional
        If provided, only these genes are used as background; otherwise all
        non‑target genes are used.
    match_expression : bool, default=False
        If True, restrict background genes to those within ±0.5 log‑expression
        units of the target median expression.
    n_components : int, default=50
        Number of PCA/LSI components.
    alternative : str, default='two-sided'
        Alternative for Mann‑Whitney U: 'two-sided', 'greater', or 'less'.
    save_plot : str, optional
        Path to save the violin plot.
    pca, tfidf, lsi : str, Path, or pre‑fitted object, optional
        Pre‑fitted transformers from training for consistent latent space.
        If not provided, they are fitted on the current data (with a warning).

    Returns
    -------
    dict
        p_value, target_lags, background_lags, mean_target, mean_background.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu
    from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

    adata.var_names_make_unique()

    if 'pseudotime' not in adata.obs:
        raise KeyError("'pseudotime' missing")
    adata = adata[adata.obs['pseudotime'].argsort()].copy()

    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Load or fit transformers
    pca, tfidf, lsi, rna_pca, atac_lsi = _load_or_fit_transformers(
        adata, pca, tfidf, lsi, n_components
    )

    # CCF surface (same as in the command‑line script)
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_lsi.T,
        rna_series=rna_pca.T,
        pseudotime=adata.obs['pseudotime'].values,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw,
                                                   return_uncertainty=True)
    comp_lag_ccf = delta_tau_smooth[:, -1]
    if adata.uns.get('pseudotime_reversed', False):
        comp_lag_ccf = -comp_lag_ccf

    # Weighted gene lags from CCF
    loadings = pca.components_
    abs_loadings = np.abs(loadings)
    norm_loadings = abs_loadings / (abs_loadings.sum(axis=0, keepdims=True) + 1e-8)
    gene_lag_ccf = (norm_loadings.T @ comp_lag_ccf).flatten()
    gene_lag_ccf = pd.Series(gene_lag_ccf, index=adata.var_names)

    # Target genes
    with open(target_genes_file, 'r') as f:
        target_genes = [line.strip() for line in f if line.strip()]
    target_genes = [g for g in target_genes if g in adata.var_names]
    print(f"Target genes found: {len(target_genes)}")
    if len(target_genes) == 0:
        raise ValueError("No target genes found in dataset.")

    # Background genes: all non‑target genes
    all_genes = set(adata.var_names)
    if background_genes_file:
        with open(background_genes_file, 'r') as f:
            bg_genes = [line.strip() for line in f if line.strip()]
        bg_genes = [g for g in bg_genes if g in adata.var_names]
    else:
        bg_genes = list(all_genes - set(target_genes))

    # Expression matching (optional)
    if match_expression:
        print("Matching expression...")
        expr = np.array(adata.X.mean(axis=0)).flatten()
        expr_series = pd.Series(expr, index=adata.var_names)
        target_expr = expr_series[target_genes].values
        target_expr_median = np.median(target_expr)
        bg_genes = [g for g in bg_genes if abs(expr_series[g] - target_expr_median) < 0.5]
        print(f"Expression‑matched background genes: {len(bg_genes)}")

    # Extract lags (no subsampling)
    target_lags = [gene_lag_ccf[g] for g in target_genes]
    bg_lags = [gene_lag_ccf[g] for g in bg_genes]

    stat, p = mannwhitneyu(target_lags, bg_lags, alternative=alternative)
    print(f"\n=== CCF‑based Δτ validation ===")
    print(f"Target Δτ mean: {np.mean(target_lags):.6f}, background Δτ mean: {np.mean(bg_lags):.6f}")
    print(f"Mann‑Whitney U: {stat}, p = {p:.4e}")

    if save_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        parts = ax.violinplot([target_lags, bg_lags], positions=[1, 2],
                              showmeans=True, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(['lightblue', 'lightcoral'][i])
            pc.set_alpha(0.7)
        for i, vals in enumerate([target_lags, bg_lags]):
            x = np.random.normal(i+1, 0.04, size=len(vals))
            ax.scatter(x, vals, alpha=0.3, s=5, color='black')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Target', 'Background'])
        ax.set_ylabel("Δτ (pseudotime units)")
        ax.set_title(f"ChIP‑seq validation: p = {p:.3e}")
        plt.tight_layout()
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        plt.close()

    return {
        'p_value': p,
        'target_lags': target_lags,
        'background_lags': bg_lags,
        'mean_target': np.mean(target_lags),
        'mean_background': np.mean(bg_lags),
    }