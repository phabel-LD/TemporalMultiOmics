#!/usr/bin/env python
"""Validate TMO using a list of target genes (aggregated δΔτ).

Uses the same PCA/LSI space as the training run.
"""

import argparse
import torch
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import mannwhitneyu
from pathlib import Path
import pickle

from tmo.models import TMOLatentModelAsymmetric

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="Combined AnnData (same used for training)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Trained asymmetric model (.pt)")
    parser.add_argument("--control_label", type=str, default="NTC")
    parser.add_argument("--perturb_label", type=str, required=True)
    parser.add_argument("--target_genes_file", type=str, required=True,
                        help="Text file with one gene symbol per line")
    parser.add_argument("--output_dir", type=str, default="./tmo_results_multip")
    parser.add_argument("--n_components", type=int, default=50)
    # Optional: paths to saved PCA/LSI objects from training
    parser.add_argument("--pca_path", type=str, default=None,
                        help="Pickle file containing the PCA object fitted on training data")
    parser.add_argument("--tfidf_path", type=str, default=None,
                        help="Pickle file containing the TfidfTransformer fitted on training ATAC")
    parser.add_argument("--lsi_path", type=str, default=None,
                        help="Pickle file containing the TruncatedSVD fitted on training ATAC")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    adata = ad.read_h5ad(args.data_path)
    control = adata[adata.obs['guide_target'] == args.control_label]
    perturbed = adata[adata.obs['guide_target'] == args.perturb_label]
    print(f"Control cells: {control.n_obs}, Perturbed cells: {perturbed.n_obs}")

    # 2. Normalize
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    rna_dense = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    atac_mat = adata.obsm['ATAC_gene']

    # 3. Load or compute PCA / LSI objects
    if args.pca_path and Path(args.pca_path).exists():
        with open(args.pca_path, 'rb') as f:
            pca = pickle.load(f)
        print("Loaded PCA from training.")
    else:
        print("Warning: PCA not provided, fitting on all cells (may not match training space).")
        pca = PCA(n_components=args.n_components, random_state=42)
        pca.fit(rna_dense)

    if args.tfidf_path and Path(args.tfidf_path).exists():
        with open(args.tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        print("Loaded TF‑IDF from training.")
    else:
        print("Warning: TF‑IDF not provided, fitting on all cells.")
        tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
        tfidf.fit(atac_mat)

    if args.lsi_path and Path(args.lsi_path).exists():
        with open(args.lsi_path, 'rb') as f:
            lsi = pickle.load(f)
        print("Loaded LSI from training.")
    else:
        print("Warning: LSI not provided, fitting on all cells.")
        lsi = TruncatedSVD(n_components=args.n_components, random_state=42)
        lsi.fit(tfidf.transform(atac_mat))

    # 4. Transform all cells
    rna_pca = pca.transform(rna_dense)
    atac_tfidf = tfidf.transform(atac_mat)
    atac_lsi = lsi.transform(atac_tfidf)

    # 5. Split indices
    ctrl_idx = [adata.obs_names.get_loc(b) for b in control.obs_names]
    pert_idx = [adata.obs_names.get_loc(b) for b in perturbed.obs_names]

    rna_ctrl = rna_pca[ctrl_idx]
    rna_pert = rna_pca[pert_idx]
    atac_ctrl = atac_lsi[ctrl_idx]
    atac_pert = atac_lsi[pert_idx]
    pt_ctrl = control.obs['pseudotime'].values
    pt_pert = perturbed.obs['pseudotime'].values

    # 6. Load model
    device = torch.device("cpu")
    model = TMOLatentModelAsymmetric(
        n_rna_components=args.n_components,
        n_atac_components=args.n_components,
        d_model=64, n_heads=4, num_layers=2, dropout=0.1,
    )
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 7. Predict lags (use_bias=True, same as training)
    with torch.no_grad():
        out_ctrl = model(torch.tensor(rna_ctrl, dtype=torch.float32),
                         torch.tensor(atac_ctrl, dtype=torch.float32),
                         torch.tensor(pt_ctrl, dtype=torch.float32),
                         use_bias=True)
        out_pert = model(torch.tensor(rna_pert, dtype=torch.float32),
                         torch.tensor(atac_pert, dtype=torch.float32),
                         torch.tensor(pt_pert, dtype=torch.float32),
                         use_bias=True)
        lag_ctrl = out_ctrl['lag_per_token'].numpy()[:, :args.n_components].mean(axis=0)
        lag_pert = out_pert['lag_per_token'].numpy()[:, :args.n_components].mean(axis=0)

    delta_tau_shift = np.abs(lag_pert - lag_ctrl)

    # 8. Map target genes to components
    with open(args.target_genes_file, 'r') as f:
        target_genes = [line.strip() for line in f if line.strip()]

    loadings = pca.components_   # (n_components, n_genes)
    target_comps = []
    for gene in target_genes:
        if gene not in adata.var_names:
            print(f"Warning: {gene} not found, skipping.")
            continue
        gene_idx = list(adata.var_names).index(gene)
        comp_weights = np.abs(loadings[:, gene_idx])
        best_comp = np.argmax(comp_weights)
        target_comps.append(best_comp)
        print(f"{gene} -> component {best_comp} (weight {comp_weights[best_comp]:.4f})")

    if not target_comps:
        raise ValueError("No target genes found in dataset.")

    target_shifts = delta_tau_shift[target_comps]
    background_shifts = np.delete(delta_tau_shift, target_comps)

    stat, p = mannwhitneyu(target_shifts, background_shifts, alternative='greater')
    print(f"\nTarget shifts (mean): {target_shifts.mean():.4f}")
    print(f"Background shifts (mean): {background_shifts.mean():.4f}")
    print(f"p-value (target > background): {p:.4e}")

    # 9. Save results
    with open(output_dir / "validation_multigene_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Control: {args.control_label}\n")
        f.write(f"Perturbation: {args.perturb_label}\n")
        f.write(f"Target genes: {', '.join(target_genes)}\n")
        f.write(f"Target components: {target_comps}\n")
        f.write(f"Target shifts mean: {target_shifts.mean():.4f}\n")
        f.write(f"Background shifts mean: {background_shifts.mean():.4f}\n")
        f.write(f"Mann‑Whitney p: {p:.4e}\n")

    print(f"Results saved to {output_dir / 'validation_multigene_results.txt'}")

if __name__ == "__main__":
    main()