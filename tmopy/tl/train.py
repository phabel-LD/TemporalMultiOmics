"""Training of asymmetric and symmetric TMO models.

Mirrors the validated command‑line pipeline: full‑dataset training,
sliding‑window CCF target (last window), joint optimisation, pickles saved.
"""

import torch
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import pickle

from tmo.models import TMOLatentModelAsymmetric
from tmo.training.losses import mse_loss
from tmo.ccf import local_ccf_surface, smooth_delta_tau_surface
from scipy.stats import spearmanr


def _compute_ccf_target(atac_lsi, rna_pca, pseudotime, pt_reversed=False):
    """Sliding‑window CCF → smoothed → last‑window target."""
    window_centers = np.linspace(0.1, 0.9, 11)
    delta_tau_raw, _, _ = local_ccf_surface(
        atac_series=atac_lsi.T,
        rna_series=rna_pca.T,
        pseudotime=pseudotime,
        window_centers=window_centers,
        window_half_width=0.1,
        max_lag=0.3,
        n_lags=31,
        bootstrap_samples=0,
    )
    delta_tau_smooth, _ = smooth_delta_tau_surface(
        window_centers, delta_tau_raw, return_uncertainty=True
    )
    ccf_target = delta_tau_smooth[:, -1]   # last window
    if pt_reversed:
        ccf_target = -ccf_target
    return ccf_target


def _compute_lcs(pred_lags, target_lags):
    valid = ~np.isnan(target_lags)
    if valid.sum() < 2:
        return np.nan
    return spearmanr(pred_lags[valid], target_lags[valid])[0]


def train_asymmetric(
    adata: ad.AnnData,
    epochs: int = 50,
    val_interval: int = 5,
    lambda_lag: float = 0.1,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
    save_model: str = None,
    output_dir: str = './tmo_results',
) -> TMOLatentModelAsymmetric:
    """
    Train asymmetric TMO model on the full dataset.

    Parameters
    ----------
    adata : AnnData
        Must contain .obs['pseudotime'] and .obsm['ATAC_gene'].
    epochs : int (default 50)
    val_interval : int (default 5) – compute LCS every N epochs.
    lambda_lag : float (default 0.1)
    batch_size : int (default 32)
    lr : float (default 1e-3)
    device : str (default 'cpu')
    save_model : str or None – path to save the best model checkpoint.
    output_dir : str – directory for checkpoints, metrics, and pickles.

    Returns
    -------
    TMOLatentModelAsymmetric (best checkpoint loaded)
    """
    # Ensure sorted by pseudotime
    if not np.all(np.diff(adata.obs['pseudotime'].values) >= 0):
        adata = adata[adata.obs['pseudotime'].argsort()].copy()

    pt_reversed = adata.uns.get('pseudotime_reversed', False)

    # Normalise RNA
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

    # Save transformers
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(output_path / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(output_path / "lsi.pkl", "wb") as f:
        pickle.dump(lsi, f)

    # CCF target (full data, sliding‑window, last window)
    pt = adata.obs['pseudotime'].values
    ccf_target = _compute_ccf_target(atac_lsi, rna_pca, pt, pt_reversed)
    ccf_target_t = torch.tensor(ccf_target, dtype=torch.float32)
    valid_mask = ~torch.isnan(ccf_target_t)

    # Tensors
    rna = torch.tensor(rna_pca, dtype=torch.float32)
    atac = torch.tensor(atac_lsi, dtype=torch.float32)
    pt_t = torch.tensor(pt, dtype=torch.float32)

    # Model
    model = TMOLatentModelAsymmetric(
        n_rna_components=n_components,
        n_atac_components=n_components,
        d_model=64, n_heads=4, num_layers=2, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_lcs = -np.inf
    best_state = None
    n_batches = (rna.shape[0] + batch_size - 1) // batch_size

    train_losses = []
    lcs_history = []

    for epoch in range(epochs):
        model.train()
        recon_sum = 0.0
        lag_sum = 0.0
        perm = torch.randperm(rna.shape[0])
        for i in range(0, rna.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            rna_b = rna[idx].to(device)
            atac_b = atac[idx].to(device)
            pt_b = pt_t[idx].to(device)

            optimizer.zero_grad()
            out = model(rna_b, atac_b, pt_b, use_bias=True)

            loss_recon = mse_loss(out['rna_pred'], rna_b)
            pred_lag_atac = out['lag_per_token'][:, :n_components].mean(dim=0)
            loss_lag = mse_loss(pred_lag_atac[valid_mask],
                                ccf_target_t[valid_mask].to(device))
            total = loss_recon + lambda_lag * loss_lag
            total.backward()
            optimizer.step()

            recon_sum += loss_recon.item()
            lag_sum += loss_lag.item()

        scheduler.step()

        avg_recon = recon_sum / n_batches
        avg_lag = lag_sum / n_batches
        total_loss = avg_recon + lambda_lag * avg_lag
        train_losses.append(total_loss)

        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out_all = model(rna.to(device), atac.to(device), pt_t.to(device),
                                use_bias=True)
                pred = out_all['lag_per_token'][:, :n_components].mean(dim=0).cpu().numpy()
                lcs = _compute_lcs(pred, ccf_target)
            lcs_val = lcs if not np.isnan(lcs) else 0.0
            lcs_history.append((epoch + 1, lcs_val, total_loss))
            print(f"Epoch {epoch+1:2d} | recon {avg_recon:.4f}  "
                  f"lag {avg_lag:.4f}  LCS {lcs_val:.4f}")

            if lcs_val > best_lcs:
                best_lcs = lcs_val
                best_state = model.state_dict().copy()
                if save_model:
                    torch.save(best_state, save_model)
                print(f"  -> Best model saved (LCS={best_lcs:.4f})")

    model.load_state_dict(best_state)
    if save_model and best_state is not None:
        torch.save(best_state, save_model)
    print(f"Training finished. Best LCS: {best_lcs:.4f}")

    # Save metrics
    df = pd.DataFrame(lcs_history, columns=["epoch", "lcs", "loss"])
    df.to_csv(output_path / "training_metrics_tmo_asymmetric.csv", index=False)
    return model


def train_symmetric(
    adata: ad.AnnData,
    epochs: int = 50,
    val_interval: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = 'cpu',
    save_model: str = None,
    output_dir: str = './tmo_results',
) -> TMOLatentModelAsymmetric:
    """
    Train symmetric baseline (identical architecture, no attention bias, no lag loss).

    Parameters
    ----------
    adata : AnnData
        Must contain .obs['pseudotime'] and .obsm['ATAC_gene'].
    epochs : int (default 50)
    val_interval : int (default 5)
    batch_size : int (default 32)
    lr : float (default 1e-3)
    device : str (default 'cpu')
    save_model : str or None
    output_dir : str

    Returns
    -------
    TMOLatentModelAsymmetric (best checkpoint loaded)
    """
    if not np.all(np.diff(adata.obs['pseudotime'].values) >= 0):
        adata = adata[adata.obs['pseudotime'].argsort()].copy()

    pt_reversed = adata.uns.get('pseudotime_reversed', False)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

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

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(output_path / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(output_path / "lsi.pkl", "wb") as f:
        pickle.dump(lsi, f)

    pt = adata.obs['pseudotime'].values
    # Still compute CCF target so that LCS can be reported (but it is not used in loss)
    ccf_target = _compute_ccf_target(atac_lsi, rna_pca, pt, pt_reversed)
    ccf_target_t = torch.tensor(ccf_target, dtype=torch.float32)

    rna = torch.tensor(rna_pca, dtype=torch.float32)
    atac = torch.tensor(atac_lsi, dtype=torch.float32)
    pt_t = torch.tensor(pt, dtype=torch.float32)

    model = TMOLatentModelAsymmetric(
        n_rna_components=n_components,
        n_atac_components=n_components,
        d_model=64, n_heads=4, num_layers=2, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_lcs = -np.inf
    best_state = None
    n_batches = (rna.shape[0] + batch_size - 1) // batch_size
    lcs_history = []

    for epoch in range(epochs):
        model.train()
        recon_sum = 0.0
        perm = torch.randperm(rna.shape[0])
        for i in range(0, rna.shape[0], batch_size):
            idx = perm[i:i+batch_size]
            rna_b = rna[idx].to(device)
            atac_b = atac[idx].to(device)
            pt_b = pt_t[idx].to(device)

            optimizer.zero_grad()
            out = model(rna_b, atac_b, pt_b, use_bias=False)

            loss_recon = mse_loss(out['rna_pred'], rna_b)
            total = loss_recon   # no lag loss
            total.backward()
            optimizer.step()
            recon_sum += loss_recon.item()

        scheduler.step()
        avg_recon = recon_sum / n_batches

        if (epoch + 1) % val_interval == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                out_all = model(rna.to(device), atac.to(device), pt_t.to(device),
                                use_bias=False)
                pred = out_all['lag_per_token'][:, :n_components].mean(dim=0).cpu().numpy()
                lcs = _compute_lcs(pred, ccf_target)
            lcs_val = lcs if not np.isnan(lcs) else 0.0
            lcs_history.append((epoch + 1, lcs_val, avg_recon))
            print(f"Epoch {epoch+1:2d} | recon {avg_recon:.4f}  LCS {lcs_val:.4f}")

            if lcs_val > best_lcs:
                best_lcs = lcs_val
                best_state = model.state_dict().copy()
                if save_model:
                    torch.save(best_state, save_model)
                print(f"  -> Best model saved (LCS={best_lcs:.4f})")

    model.load_state_dict(best_state)
    if save_model and best_state is not None:
        torch.save(best_state, save_model)
    print(f"Training finished. Best LCS: {best_lcs:.4f}")

    df = pd.DataFrame(lcs_history, columns=["epoch", "lcs", "loss"])
    df.to_csv(output_path / "training_metrics_tmo_symmetric.csv", index=False)
    return model