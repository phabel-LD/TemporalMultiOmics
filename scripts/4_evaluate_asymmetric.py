#!/usr/bin/env python
"""Evaluate trained asymmetric TMO model: Δτ heatmap, clustering, LCS, and annotation.

Corrected clustering to handle constant / NaN component profiles.
"""

import argparse
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import pdist
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

from tmo.models import TMOLatentModelAsymmetric
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface
from tmo.plotting import plot_delta_tau_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./tmo_results/tmo_asymmetric_best.pt")
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--n_cells", type=int, default=None, help="Number of cells to use (all if None)")
    parser.add_argument("--annotation_mode", type=str, default="component",
                        choices=["component", "cluster_deltatau", "cluster_aggregate"],
                        help="How to label rows")
    parser.add_argument("--dist_threshold", type=float, default=0.8,
                        help="Distance threshold for clustering")
    parser.add_argument("--standardize", action="store_true", default=True,
                        help="Standardize Δτ profiles before clustering")
    parser.add_argument("--use_windowed_sparsity", action="store_true",
                        help="Enable windowed sparse attention for large datasets")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data and compute latent representations
    # ------------------------------------------------------------------
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    if 'pseudotime' not in adata.obs:
        raise KeyError("'pseudotime' missing in .obs")
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
    if args.n_cells is not None and args.n_cells < adata.n_obs:
        adata = adata[:args.n_cells].copy()
        print(f"Using first {args.n_cells} cells")

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

    # ------------------------------------------------------------------
    # 2. CCF Δτ surface
    # ------------------------------------------------------------------
    atac_comp_T = atac_lsi.T
    rna_comp_T = rna_pca.T
    pseudotime_arr = adata.obs['pseudotime'].values
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_comp_T,
        rna_series=rna_comp_T,
        pseudotime=pseudotime_arr,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)

    # Flip sign if dataset was reversed
    if adata.uns.get('pseudotime_reversed', False):
        delta_tau_smooth = -delta_tau_smooth
        print("Flipped sign of Δτ to maintain uniform interpretation (positive = ATAC leads).")
    else:
        print("Dataset not reversed; Δτ signs kept as computed.")

    # Write reversal flag to a text file
    with open(output_dir / "pseudotime_reversed_flag.txt", "w") as f:
        f.write(str(adata.uns.get('pseudotime_reversed', False)))

    # ------------------------------------------------------------------
    # 3. Load component annotations
    # ------------------------------------------------------------------
    annot_file = output_dir / "component_annotation.csv"
    if annot_file.exists():
        annot_df = pd.read_csv(annot_file)
        comp_to_label = {}
        for _, row in annot_df.iterrows():
            comp = row['Component']
            go_term = row['GO_terms'].split(';')[0].strip() if pd.notna(row['GO_terms']) else ""
            if go_term:
                comp_to_label[comp] = f"{comp} ({go_term[:30]})"
            else:
                comp_to_label[comp] = comp
    else:
        comp_to_label = {f"Comp{i}": f"Comp{i}" for i in range(n_atac)}
        print("Annotation file not found; using generic labels.")
        annot_df = None

    comp_names = [comp_to_label.get(f"Comp{i}", f"Comp{i}") for i in range(n_atac)]

    # ------------------------------------------------------------------
    # 4. Prepare heatmap data and labels
    # ------------------------------------------------------------------
    if args.annotation_mode == "component":
        dt_heatmap = delta_tau_smooth
        row_labels = comp_names
    else:
        # --- clustering with robustness against constant / NaN components ---
        dt_cluster = delta_tau_smooth.copy()

        # keep only valid rows
        valid_rows = np.all(np.isfinite(dt_cluster), axis=1) & (dt_cluster.std(axis=1) > 1e-12)
        dt_valid = dt_cluster[valid_rows, :]
        valid_idx = np.where(valid_rows)[0]   # original component indices
        print(f"Retained {len(valid_idx)} / {dt_cluster.shape[0]} components for clustering "
              f"({dt_cluster.shape[0] - len(valid_idx)} removed due to constant / NaN profiles).")

        if args.standardize:
            row_mean = dt_valid.mean(axis=1, keepdims=True)
            row_std = dt_valid.std(axis=1, keepdims=True)
            dt_valid = (dt_valid - row_mean) / (row_std + 1e-8)
            print("Standardized Δτ profiles for clustering.")

        dist = pdist(dt_valid, metric='correlation')

        # safety: replace any remaining non‑finite distances
        if not np.all(np.isfinite(dist)):
            print("Warning: non‑finite distances found; replacing with max distance (1.0).")
            dist = np.where(np.isfinite(dist), dist, 1.0)

        # hierarchical clustering on the valid set
        row_linkage = linkage(dist, method='ward')
        cluster_ids_valid = fcluster(row_linkage, args.dist_threshold, criterion='distance')
        n_clusters = len(np.unique(cluster_ids_valid))
        print(f"Number of clusters at threshold {args.dist_threshold}: {n_clusters}")

        # map back to full component space (invalid ones get their own cluster -1, but we'll handle them separately)
        full_cluster = np.full(delta_tau_smooth.shape[0], -1, dtype=int)
        full_cluster[valid_idx] = cluster_ids_valid

        if args.annotation_mode == "cluster_deltatau":
            # reorder valid components by dendrogram, append invalid ones at the top
            row_order = leaves_list(row_linkage)   # ordering within valid set
            ordered_idx = valid_idx[row_order]     # original indices of valid components in cluster order
            # add any invalid components at the end (or beginning)
            invalid_idx = np.where(~valid_rows)[0]
            final_order = np.concatenate([ordered_idx, invalid_idx])
            dt_heatmap = delta_tau_smooth[final_order, :]
            row_labels = [comp_names[i] for i in final_order]
        else:  # cluster_aggregate
            cluster_median = []
            cluster_label_list = []
            for cl in np.unique(cluster_ids_valid):
                comp_indices = np.where(full_cluster == cl)[0]
                median_dt = np.median(delta_tau_smooth[comp_indices, :], axis=0)
                cluster_median.append(median_dt)

                # representative GO term
                go_terms = []
                for idx in comp_indices:
                    comp_name_f = comp_names[idx].split(' (')[0]
                    if annot_df is not None:
                        row = annot_df[annot_df['Component'] == comp_name_f]
                        if not row.empty:
                            go_str = row['GO_terms'].values[0]
                            if pd.notna(go_str):
                                go_terms.append(go_str.split(';')[0].strip())
                if go_terms:
                    most_common = Counter(go_terms).most_common(1)[0][0]
                    cluster_label_list.append(f"Cluster {cl} ({most_common})")
                else:
                    cluster_label_list.append(f"Cluster {cl}")
            
            # Sort valid clusters
            if len(cluster_median) > 1:
                median_dist = pdist(np.array(cluster_median), metric='correlation')
                if not np.all(np.isfinite(median_dist)):
                    median_dist = np.where(np.isfinite(median_dist), median_dist, 1.0)
                median_linkage = linkage(median_dist, method='ward')
                reorder = leaves_list(median_linkage)
                cluster_median = [cluster_median[i] for i in reorder]
                cluster_label_list = [cluster_label_list[i] for i in reorder]
            
            # if there are invalid components, we can add them as a single "No cluster" row
            if not np.all(valid_rows):
                invalid_idx = np.where(~valid_rows)[0]
                median_invalid = np.median(delta_tau_smooth[invalid_idx, :], axis=0)
                cluster_median.append(median_invalid)
                cluster_label_list.append("No cluster (constant/NaN)")
            dt_heatmap = np.array(cluster_median)
            row_labels = cluster_label_list

    # ------------------------------------------------------------------
    # 5. Plot Δτ heatmap
    # ------------------------------------------------------------------
    fig1 = plot_delta_tau_heatmap(
        dt_heatmap,
        window_centers,
        row_labels,
        title="Regulatory lag Δτ(τ)",
        save_path=str(output_dir / "fig1_latent_delta_tau.pdf"),
    )
    plt.close(fig1)
    print("Saved latent Δτ heatmap (Figure 1).")

    # ------------------------------------------------------------------
    # 6. Load model and compute LCS (full data)
    # ------------------------------------------------------------------
    print("Loading model...")
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_pca,
        n_atac_components=n_atac,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        use_windowed_sparsity=args.use_windowed_sparsity,
    )
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
        model.eval()
        with torch.no_grad():
            rna_t = torch.tensor(rna_pca, dtype=torch.float32).to(device)
            atac_t = torch.tensor(atac_lsi, dtype=torch.float32).to(device)
            pt_t = torch.tensor(pseudotime_arr, dtype=torch.float32).to(device)
            out = model(rna_t, atac_t, pt_t)
            pred_lags = out['lag_per_token'].cpu().numpy().mean(axis=0)[:n_atac]
            ccf_target = delta_tau_smooth[:, -1]
            valid = ~np.isnan(ccf_target)
            if valid.sum() > 1:
                lcs, _ = spearmanr(pred_lags[valid], ccf_target[valid])
                print(f"Lag Concordance Score (latent components): {lcs:.4f}")
            else:
                print("Not enough valid components for LCS.")
    else:
        print("Model not found; skipping LCS.")

    print("Evaluation finished. Figures saved in", output_dir)


if __name__ == "__main__":
    main()