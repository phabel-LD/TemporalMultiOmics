#!/usr/bin/env python
"""Train symmetric TMO latent model (no attention bias) as a baseline.

Corrected version:
- Train/validation split before PCA/LSI
- Independent CCF targets for training and validation
- Pass‑1 reconstruction loss (same architecture as asymmetric, just no bias)
- Pseudotime reversal flag handled
"""

import argparse
import torch
import numpy as np
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from tmo.models import TMOLatentModelAsymmetric
from tmo.training.losses import mse_loss
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

def compute_lcs_from_predictions(pred_lags, target_lags, valid_mask=None):
    if valid_mask is None:
        valid_mask = ~np.isnan(target_lags)
    if np.sum(valid_mask) < 2:
        return np.nan
    return spearmanr(pred_lags[valid_mask], target_lags[valid_mask])[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_pass1", type=float, default=0.1)
    parser.add_argument("--lambda_lag", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "tmo_symmetric"
    print(f"Training {model_name} on {device}")

    # 1. Load and split (same as asymmetric)
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    adata = adata[adata.obs['pseudotime'].argsort()].copy()

    pt_reversed = adata.uns.get('pseudotime_reversed', False)
    if pt_reversed:
        print("Pseudotime was reversed; CCF signs will be flipped for training.")

    n_cells = adata.n_obs
    indices = np.arange(n_cells)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n_cells)
    train_idx = indices[:split]
    val_idx = indices[split:]

    adata_train = adata[train_idx].copy()
    adata_train = adata_train[adata_train.obs['pseudotime'].argsort()].copy()
    adata_val = adata[val_idx].copy()
    adata_val = adata_val[adata_val.obs['pseudotime'].argsort()].copy()

    sc.pp.normalize_total(adata_train, target_sum=1e4)
    sc.pp.log1p(adata_train)
    sc.pp.normalize_total(adata_val, target_sum=1e4)
    sc.pp.log1p(adata_val)

    # 2. PCA on training RNA
    n_pca = 50
    pca = PCA(n_components=n_pca, random_state=42)
    rna_train = pca.fit_transform(adata_train.X.toarray() if hasattr(adata_train.X, 'toarray') else adata_train.X)
    rna_val = pca.transform(adata_val.X.toarray() if hasattr(adata_val.X, 'toarray') else adata_val.X)

    # 3. LSI on training ATAC
    atac_train_mat = adata_train.obsm['ATAC_gene']
    atac_val_mat = adata_val.obsm['ATAC_gene']
    if not hasattr(atac_train_mat, 'toarray'):
        atac_train_mat = atac_train_mat.toarray()
        atac_val_mat = atac_val_mat.toarray()

    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_train_tfidf = tfidf.fit_transform(atac_train_mat)
    atac_val_tfidf = tfidf.transform(atac_val_mat)

    n_atac = 50
    lsi = TruncatedSVD(n_components=n_atac, random_state=42)
    atac_train_lsi = lsi.fit_transform(atac_train_tfidf)
    atac_val_lsi = lsi.transform(atac_val_tfidf)

    # 4. CCF target on training set only
    print("Computing CCF on training set...")
    pt_train = adata_train.obs['pseudotime'].values
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_train_lsi.T,
        rna_series=rna_train.T,
        pseudotime=pt_train,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw, return_uncertainty=True)
    ccf_target_train = delta_tau_smooth[:, -1]
    if pt_reversed:
        ccf_target_train = -ccf_target_train
    ccf_target_train = torch.tensor(ccf_target_train, dtype=torch.float32)
    valid_mask_train = ~torch.isnan(ccf_target_train)

    # CCF target for validation
    print("Computing CCF on validation set...")
    pt_val = adata_val.obs['pseudotime'].values
    delta_tau_raw_val, _, _ = local_ccf_surface(
        atac_series=atac_val_lsi.T,
        rna_series=rna_val.T,
        pseudotime=pt_val,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth_val, _ = smooth_delta_tau_surface(window_centers, delta_tau_raw_val, return_uncertainty=True)
    ccf_target_val = delta_tau_smooth_val[:, -1].copy()
    if pt_reversed:
        ccf_target_val = -ccf_target_val

    # === Save fitted transformers for later reuse (e.g., Perturb-seq validation) ===
    import pickle
    with open(output_dir / "pca.pkl", "wb") as f_pca:
        pickle.dump(pca, f_pca)
    with open(output_dir / "tfidf.pkl", "wb") as f_tfidf:
        pickle.dump(tfidf, f_tfidf)
    with open(output_dir / "lsi.pkl", "wb") as f_lsi:
        pickle.dump(lsi, f_lsi)
    print("Saved PCA, TF‑IDF, and LSI objects for validation reuse.")

    # 5. Tensors
    rna_train_t = torch.tensor(rna_train, dtype=torch.float32)
    atac_train_t = torch.tensor(atac_train_lsi, dtype=torch.float32)
    pt_train_t = torch.tensor(pt_train, dtype=torch.float32)
    rna_val_t = torch.tensor(rna_val, dtype=torch.float32)
    atac_val_t = torch.tensor(atac_val_lsi, dtype=torch.float32)
    pt_val_t = torch.tensor(pt_val, dtype=torch.float32)

    # 6. Model
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_pca,
        n_atac_components=n_atac,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_lcs = -np.inf
    best_model_path = output_dir / f"{model_name}_best.pt"
    n_batches = (len(train_idx) + args.batch_size - 1) // args.batch_size
    train_losses = []
    val_lcs_list = []
    epoch_list = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_recon = 0.0
        epoch_loss_pass1 = 0.0
        epoch_loss_lag = 0.0
        perm = torch.randperm(len(train_idx))
        for i in range(0, len(train_idx), args.batch_size):
            idx = perm[i:i+args.batch_size]
            rna_batch = rna_train_t[idx].to(device)
            atac_batch = atac_train_t[idx].to(device)
            pt_batch = pt_train_t[idx].to(device)

            optimizer.zero_grad()
            # Symmetric: use_bias=False → no attention bias, but still predicts lags
            out = model(rna_batch, atac_batch, pt_batch, use_bias=False)

            loss_recon = mse_loss(out['rna_pred'], rna_batch)
            loss_pass1 = mse_loss(out['rna_pred_pass1'], rna_batch)

            n_atac_cur = atac_batch.shape[1]
            pred_lag_atac = out['lag_per_token'][:, :n_atac_cur].mean(dim=0)
            loss_lag = mse_loss(pred_lag_atac[valid_mask_train],
                                ccf_target_train[valid_mask_train].to(device)) / 0.01

            total = (args.lambda_recon * loss_recon +
                     args.lambda_pass1 * loss_pass1 +
                     args.lambda_lag * loss_lag)
            total.backward()
            optimizer.step()

            epoch_loss_recon += loss_recon.item()
            epoch_loss_pass1 += loss_pass1.item()
            epoch_loss_lag += loss_lag.item()

        scheduler.step()

        if (epoch + 1) % args.val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(rna_val_t.to(device), atac_val_t.to(device),
                                pt_val_t.to(device), use_bias=False)
                pred_val = val_out['lag_per_token'].cpu().numpy().mean(axis=0)
                pred_val_atac = pred_val[:n_atac]
                lcs = compute_lcs_from_predictions(pred_val_atac, ccf_target_val)
            val_lcs_list.append(lcs if not np.isnan(lcs) else 0.0)
            epoch_list.append(epoch + 1)
            avg_recon = epoch_loss_recon / n_batches
            avg_pass1 = epoch_loss_pass1 / n_batches
            avg_lag = epoch_loss_lag / n_batches
            print(f"Epoch {epoch+1}/{args.epochs} | Recon: {avg_recon:.4f} "
                  f"Pass1: {avg_pass1:.4f} Lag: {avg_lag:.4f} | LCS: {lcs:.4f}")
            if lcs > best_lcs:
                best_lcs = lcs
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Best model saved (LCS={best_lcs:.4f})")

        train_losses.append(epoch_loss_recon / n_batches)

    torch.save(model.state_dict(), output_dir / f"{model_name}_last.pt")
    print(f"Training finished. Best LCS: {best_lcs:.4f}")

    import pandas as pd
    full_epochs = list(range(1, args.epochs+1))
    full_lcs = [np.nan] * args.epochs
    for i, ep in enumerate(epoch_list):
        full_lcs[ep-1] = val_lcs_list[i]
    df = pd.DataFrame({'epoch': full_epochs, 'loss': train_losses, 'lcs': full_lcs})
    df.to_csv(output_dir / f"training_metrics_{model_name}.csv", index=False)

if __name__ == "__main__":
    main()