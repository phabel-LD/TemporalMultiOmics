#!/usr/bin/env python
"""Train asymmetric TMO model – full metrics, best model, final summary.

Training uses the full dataset (original pipeline) and computes in‑set LCS.
The independent biological validations remain rigorous.
"""

import argparse
import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from scipy.stats import spearmanr
from pathlib import Path
import pickle
import warnings

from tmo.models import TMOLatentModelAsymmetric
from tmo.training.losses import mse_loss
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.gaussian_process")


def compute_lcs(pred_lags, target_lags):
    """Spearman correlation, excluding NaN targets."""
    valid = ~np.isnan(target_lags)
    if valid.sum() < 2:
        return np.nan
    return spearmanr(pred_lags[valid], target_lags[valid])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tmo_results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val_interval", type=int, default=5,
                        help="Compute LCS every N epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lambda_lag", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_windowed_sparsity", action="store_true",
                        help="Enable windowed sparse attention for large datasets")
    parser.add_argument("--symmetric", action="store_true",
                        help="Train symmetric baseline (no attention bias)")
    parser.add_argument("--ablate_cell_state", action="store_true",
                        help="Zero out the cell embedding before the LagMLP (cell‑state ablation)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    device = torch.device("cpu")

    # -------------------------------------------------------------
    # 1. Load and preprocess
    # -------------------------------------------------------------
    print("Loading data...")
    adata = ad.read_h5ad(args.data_path)
    adata = adata[adata.obs['pseudotime'].argsort()].copy()
    adata.var_names_make_unique()
    pt_reversed = adata.uns.get('pseudotime_reversed', False)
    if pt_reversed:
        print("Pseudotime was reversed; CCF signs will be flipped.")

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # PCA and LSI on all cells
    n_components = 50
    pca = PCA(n_components=n_components, random_state=42)
    rna_pca = pca.fit_transform(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    )

    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # Save transformers for later validation
    with open(output_dir / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(output_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(output_dir / "lsi.pkl", "wb") as f:
        pickle.dump(lsi, f)

    # -------------------------------------------------------------
    # 2. CCF target (original sliding‑window, last‑window smoothed)
    # -------------------------------------------------------------
    print("Computing CCF surface...")
    atac_all_T = atac_lsi.T
    rna_all_T = rna_pca.T
    pt_all = adata.obs['pseudotime'].values
    window_centers = np.linspace(0.1, 0.9, 11)

    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_all_T,
        rna_series=rna_all_T,
        pseudotime=pt_all,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(
        window_centers, delta_tau_raw, return_uncertainty=True
    )
    ccf_target = delta_tau_smooth[:, -1]   # last pseudotime window

    if pt_reversed:
        ccf_target = -ccf_target

    ccf_target_t = torch.tensor(ccf_target, dtype=torch.float32)
    # Save the training CCF target for held‑out evaluation
    with open(output_dir / "ccf_target.pkl", "wb") as f:
        pickle.dump(ccf_target, f)

    valid_mask = ~torch.isnan(ccf_target_t)

    # -------------------------------------------------------------
    # 3. Prepare tensors
    # -------------------------------------------------------------
    rna = torch.tensor(rna_pca, dtype=torch.float32)
    atac = torch.tensor(atac_lsi, dtype=torch.float32)
    pt = torch.tensor(pt_all, dtype=torch.float32)

    # -------------------------------------------------------------
    # 4. Model, optimizer, scheduler
    # -------------------------------------------------------------
    model_name = "tmo_symmetric" if args.symmetric else "tmo_asymmetric"
    use_bias = not args.symmetric
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_components,
        n_atac_components=n_components,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        use_windowed_sparsity=args.use_windowed_sparsity,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_lcs = -np.inf
    best_state = None
    best_epoch = 0
    n_batches = (rna.shape[0] + args.batch_size - 1) // args.batch_size

    # -------------------------------------------------------------
    # 5. Training loop
    # -------------------------------------------------------------
    model.ablate_cell_state = args.ablate_cell_state
    lcs_history = []

    for epoch in range(args.epochs):
        model.train()
        recon_sum = 0.0
        lag_sum = 0.0
        perm = torch.randperm(rna.shape[0])
        for i in range(0, rna.shape[0], args.batch_size):
            idx = perm[i:i+args.batch_size]
            rna_b = rna[idx].to(device)
            atac_b = atac[idx].to(device)
            pt_b = pt[idx].to(device)

            optimizer.zero_grad()
            out = model(rna_b, atac_b, pt_b, use_bias=use_bias)

            loss_recon = mse_loss(out['rna_pred'], rna_b)
            if args.symmetric:
                total = loss_recon
            else:
                pred_lag_atac = out['lag_per_token'][:, :n_components].mean(dim=0)
                loss_lag = mse_loss(pred_lag_atac[valid_mask],
                                    ccf_target_t[valid_mask].to(device))
                total = loss_recon + args.lambda_lag * loss_lag
                lag_sum += loss_lag.item()
            total.backward()
            optimizer.step()

            recon_sum += loss_recon.item()

        scheduler.step()

        avg_recon = recon_sum / n_batches
        avg_lag = lag_sum / n_batches
        total_loss = avg_recon + args.lambda_lag * avg_lag

        # Compute LCS every val_interval epochs (and first epoch)
        if (epoch + 1) % args.val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out_all = model(rna.to(device), atac.to(device), pt.to(device),
                                use_bias=use_bias)
                pred = out_all['lag_per_token'][:, :n_components].mean(dim=0).cpu().numpy()
                lcs = compute_lcs(pred, ccf_target)
            lcs_val = lcs if not np.isnan(lcs) else 0.0
            lcs_history.append((epoch + 1, lcs_val, avg_recon, avg_lag, total_loss))
            print(f"Epoch {epoch+1:2d} | recon {avg_recon:.4f}  "
                  f"lag {avg_lag:.4f}  LCS {lcs_val:.4f}")

            if lcs_val > best_lcs:
                best_lcs = lcs_val
                best_state = model.state_dict().copy()
                best_epoch = epoch + 1
                torch.save(best_state, output_dir / f"{model_name}_best.pt")
                print(f"  -> Best model saved (LCS={best_lcs:.4f})")

    # -------------------------------------------------------------
    # 6. Final evaluation of the best model
    # -------------------------------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out_all = model(rna.to(device), atac.to(device), pt.to(device),
                        use_bias=use_bias)
        pred_lag_all = out_all['lag_per_token'][:, :n_components].mean(dim=0).cpu().numpy()
        final_recon = mse_loss(out_all['rna_pred'],
                               rna.to(device)).item()
        final_lag = mse_loss(
            torch.tensor(pred_lag_all[valid_mask.numpy()], dtype=torch.float32),
            ccf_target_t[valid_mask]
        ).item()
        final_lcs = compute_lcs(pred_lag_all, ccf_target)

    # -------------------------------------------------------------
    # 7. Print detailed best‑model metrics
    # -------------------------------------------------------------
    print("\n" + "="*60)
    print("Best model (epoch {}):".format(best_epoch))
    print("  Reconstruction loss: {:.6f}".format(final_recon))
    print("  Lag consistency loss: {:.6f}".format(final_lag))
    print("  LCS (Spearman)     : {:.4f}".format(final_lcs))
    print("  Mean predicted Δτ  : {:.4f}".format(np.mean(pred_lag_all)))
    print("  Std  predicted Δτ  : {:.4f}".format(np.std(pred_lag_all)))
    print("  Mean target Δτ     : {:.4f}".format(
        np.nanmean(ccf_target[valid_mask.numpy()])))
    print("  Std  target Δτ     : {:.4f}".format(
        np.nanstd(ccf_target[valid_mask.numpy()])))
    print("="*60 + "\n")

    # -------------------------------------------------------------
    # 8. Save training metrics CSV
    # -------------------------------------------------------------
    df = pd.DataFrame(lcs_history, columns=["epoch", "lcs", "recon", "lag", "loss"])
    df.to_csv(output_dir / f"training_metrics_{model_name}.csv", index=False)
    torch.save(model.state_dict(), output_dir / f"{model_name}_last.pt")
    print(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()