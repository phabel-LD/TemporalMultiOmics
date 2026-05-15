"""Evaluate trained asymmetric model: compute Δτ surface, LCS, and return data for plotting.

Now supports optional cluster‑aggregation (matching the command‑line pipeline).
"""

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
from collections import Counter
from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface
from scipy.stats import spearmanr


def _compute_lcs(pred_lags, target_lags):
    valid = ~np.isnan(target_lags)
    if valid.sum() < 2:
        return np.nan
    return spearmanr(pred_lags[valid], target_lags[valid])[0]


def evaluate_asymmetric(
    adata,
    model_path,
    n_components=50,
    cluster_aggregate=False,
    dist_threshold=1.0,
    standardize=True,
):
    """
    Compute Δτ surface, LCS, and return data for plotting.

    Parameters
    ----------
    adata : AnnData
        Must contain .obs['pseudotime'] and .obsm['ATAC_gene'].
    model_path : str
        Path to trained asymmetric model (.pt).
    n_components : int, default=50
        Number of PCA/LSI components.
    cluster_aggregate : bool, default=False
        If True, perform hierarchical clustering on the Δτ profiles and return
        cluster‑level medians and labels (matching the command‑line Figure 1).
    dist_threshold : float, default=1.0
        Distance threshold for forming flat clusters (only relevant when
        cluster_aggregate=True).
    standardize : bool, default=True
        Whether to standardize profiles before clustering.

    Returns
    -------
    delta_tau_heatmap : np.ndarray (n_components or n_clusters, n_windows)
        Δτ surface (sign‑flipped for uniform interpretation).
    window_centers : np.ndarray
        Pseudotime window centres.
    lcs_raw : float
        Lag Concordance Score (in‑set).
    row_labels : list of str (only if cluster_aggregate=True)
        Labels for each row of the heatmap.
    """
    # Ensure RNA is log‑normalised
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # Latent projections
    pca = PCA(n_components=n_components, random_state=42)
    rna_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # CCF surface
    atac_comp_T = atac_lsi.T
    rna_comp_T = rna_pca.T
    pt = adata.obs['pseudotime'].values
    if not np.all(np.diff(pt) >= 0):
        raise ValueError("pseudotime must be sorted in increasing order")
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
    delta_tau_smooth_raw, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)

    # Save raw CCF target for LCS
    ccf_target_raw = delta_tau_smooth_raw[:, -1].copy()

    # For heatmap, optionally flip signs
    delta_tau_heatmap = delta_tau_smooth_raw.copy()
    if adata.uns.get('pseudotime_reversed', False):
        delta_tau_heatmap = -delta_tau_heatmap
        print("Flipped Δτ signs for heatmap (positive = ATAC leads).")
    else:
        print("Dataset not reversed; Δτ signs kept as computed.")

    # Model predictions
    device = torch.device("cpu")
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_components,
        n_atac_components=n_components,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(rna_pca, dtype=torch.float32),
                    torch.tensor(atac_lsi, dtype=torch.float32),
                    torch.tensor(pt, dtype=torch.float32))
        pred_lags = out['lag_per_token'][:, :n_components].numpy().mean(axis=0)

    # LCS on raw values
    lcs_raw = _compute_lcs(pred_lags, ccf_target_raw)
    print(f"LCS (raw, for validation) = {lcs_raw:.4f}")

    if not cluster_aggregate:
        return delta_tau_heatmap, window_centers, lcs_raw

    # --- Cluster aggregation (matching command‑line script) ---
    dt_cluster = delta_tau_heatmap.copy()

    # Remove components with constant or invalid profiles
    valid_rows = np.all(np.isfinite(dt_cluster), axis=1) & (dt_cluster.std(axis=1) > 1e-12)
    dt_valid = dt_cluster[valid_rows, :]
    valid_idx = np.where(valid_rows)[0]
    print(f"Retained {len(valid_idx)} / {dt_cluster.shape[0]} components for clustering "
          f"({dt_cluster.shape[0] - len(valid_idx)} removed due to constant / NaN profiles).")

    if standardize:
        row_mean = dt_valid.mean(axis=1, keepdims=True)
        row_std = dt_valid.std(axis=1, keepdims=True)
        dt_valid = (dt_valid - row_mean) / (row_std + 1e-8)
        print("Standardized Δτ profiles for clustering.")

    dist = pdist(dt_valid, metric='correlation')
    if not np.all(np.isfinite(dist)):
        print("Warning: non‑finite distances found; replacing with max distance (1.0).")
        dist = np.where(np.isfinite(dist), dist, 1.0)

    row_linkage = linkage(dist, method='ward')
    cluster_ids_valid = fcluster(row_linkage, dist_threshold, criterion='distance')
    n_clusters = len(np.unique(cluster_ids_valid))
    print(f"Number of clusters at threshold {dist_threshold}: {n_clusters}")

    # Map back to full component space
    full_cluster = np.full(dt_cluster.shape[0], -1, dtype=int)
    full_cluster[valid_idx] = cluster_ids_valid

    # Collect medians and labels
    cluster_medians = []
    cluster_labels = []
    # Load component annotation if available
    annot_df = adata.uns.get('component_annotation', None)
    comp_names = []
    if annot_df is not None:
        for _, row in annot_df.iterrows():
            go_term = row['GO_terms'].split(';')[0].strip() if pd.notna(row['GO_terms']) else ""
            comp_names.append(f"{row['Component']} ({go_term[:30]})" if go_term else row['Component'])
    else:
        comp_names = [f"Comp{i}" for i in range(n_components)]

    for cl in np.unique(cluster_ids_valid):
        comp_indices = np.where(full_cluster == cl)[0]
        median_dt = np.median(dt_cluster[comp_indices, :], axis=0)
        cluster_medians.append(median_dt)

        # Representative GO term
        go_terms = []
        for idx in comp_indices:
            if annot_df is not None:
                comp_name = f"Comp{idx}"
                row = annot_df[annot_df['Component'] == comp_name]
                if not row.empty:
                    go_str = row['GO_terms'].values[0]
                    if pd.notna(go_str):
                        go_terms.append(go_str.split(';')[0].strip())
        if go_terms:
            most_common = Counter(go_terms).most_common(1)[0][0]
            cluster_labels.append(f"Cluster {cl} ({most_common})")
        else:
            cluster_labels.append(f"Cluster {cl}")

    # Sort clusters by similarity of their median profiles
    if len(cluster_medians) > 1:
        median_dist = pdist(np.array(cluster_medians), metric='correlation')
        if not np.all(np.isfinite(median_dist)):
            median_dist = np.where(np.isfinite(median_dist), median_dist, 1.0)
        median_linkage = linkage(median_dist, method='ward')
        reorder = leaves_list(median_linkage)
        cluster_medians = [cluster_medians[i] for i in reorder]
        cluster_labels = [cluster_labels[i] for i in reorder]

    # Add invalid components as a final row if present
    if not np.all(valid_rows):
        invalid_idx = np.where(~valid_rows)[0]
        median_invalid = np.median(dt_cluster[invalid_idx, :], axis=0)
        cluster_medians.append(median_invalid)
        cluster_labels.append("No cluster (constant/NaN)")

    return np.array(cluster_medians), window_centers, lcs_raw, cluster_labels