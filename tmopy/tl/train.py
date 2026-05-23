"""Training of asymmetric and symmetric TMO models.

This module provides the two main training functions used in the validated
TMO pipeline: ``train_asymmetric`` and ``train_symmetric``.  They implement
the end‑to‑end training loop that operates on the full dataset (no
train/validation split), using a sliding‑window CCF target and joint
optimisation of reconstruction and lag consistency losses.

After training, the functions save:

- The best model checkpoint (by in‑set LCS).
- Training metrics (epoch, LCS, loss) as a CSV file.
- The pre‑fitted PCA, TF‑IDF, and LSI transformers as pickle files, so
  that they can be reused by the evaluation and validation scripts.

The symmetric baseline is trained with the identical architecture but
without the attention bias (``use_bias=False``) and without the lag
consistency loss, providing a fair comparison that isolates the effect
of the asymmetric attention mechanism.
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
    """Compute the CCF‑derived training target for the lag consistency loss.

    The function performs three steps:

        1. Run the local sliding‑window cross‑correlation (CCF) on the
           provided ATAC and RNA latent components.
        2. Smooth the resulting Δτ surface with a Gaussian process.
        3. Extract the smoothed lag value at the last pseudotime window
           (τ = 0.9).  If the pseudotime was reversed during preprocessing,
           the sign of the target is flipped so that the model learns the
           biologically correct orientation (positive = ATAC leads).

    Parameters
    ----------
    atac_lsi : np.ndarray, shape (n_cells, n_components)
        ATAC latent representations (from Truncated SVD of TF‑IDF matrix).
    rna_pca : np.ndarray, shape (n_cells, n_components)
        RNA latent representations (from PCA).
    pseudotime : np.ndarray, shape (n_cells,)
        Sorted pseudotime values in [0, 1].
    pt_reversed : bool, default=False
        Whether the pseudotime was reversed during preprocessing.  If
        True, the returned target lags are multiplied by –1.

    Returns
    -------
    np.ndarray, shape (n_components,)
        The CCF target lag for each ATAC component.
    """

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
    ccf_target = delta_tau_smooth[:, -1] # last pseudotime window
    if pt_reversed:
        ccf_target = -ccf_target
    return ccf_target


def _compute_lcs(pred_lags, target_lags) -> float:
    """Compute the Lag Concordance Score (Spearman correlation).

    Components whose target lag is NaN are excluded from the calculation.
    At least two valid components are required; otherwise NaN is returned.

    Parameters
    ----------
    pred_lags : np.ndarray, shape (n_components,)
        Mean predicted lags per ATAC component (averaged over cells).
    target_lags : np.ndarray, shape (n_components,)
        CCF‑derived target lags.

    Returns
    -------
    float
        Spearman correlation, or NaN if fewer than two valid components.
    """

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
    """Train the asymmetric TMO model on the full dataset.

    The training follows the exact procedure described in the manuscript:

    - RNA is normalised (log1p) and projected to 50 PCA components.
    - ATAC is transformed with TF‑IDF and projected to 50 LSI components.
    - The CCF target is computed on the entire dataset (sliding‑window,
      last‑window smoothed).
    - The model is trained end‑to‑end with a reconstruction loss (MSE on
      the RNA latent components) and a lag consistency loss (MSE between
      predicted mean lags and the CCF targets).
    - Every ``val_interval`` epochs, the LCS is computed on the full
      dataset (in‑set diagnostic) and the best checkpoint is saved.

    The fitted PCA, TF‑IDF, and LSI objects are pickled in the output
    directory so that downstream scripts can apply the exact same
    transformations (e.g., for Perturb‑seq or ChIP‑seq validation).

    Parameters
    ----------
    adata : AnnData
        Must contain ``.obs['pseudotime']`` (sorted) and
        ``.obsm['ATAC_gene']`` (gene‑level ATAC matrix).
    epochs : int, default=50
        Total number of training epochs.
    val_interval : int, default=5
        Compute LCS and optionally save a checkpoint every N epochs.
    lambda_lag : float, default=0.1
        Weight of the lag consistency loss relative to the reconstruction
        loss (which has weight 1.0).
    batch_size : int, default=32
        Mini‑batch size.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    device : str, default='cpu'
        Device for training (``'cpu'`` or ``'cuda'``).
    save_model : str, optional
        Path where the best model checkpoint will be saved.  If not
        provided, the checkpoint is only kept in memory and can be
        accessed via the returned model.
    output_dir : str, default='./tmo_results'
        Directory for the training metrics CSV and the pickled
        transformers.

    Returns
    -------
    TMOLatentModelAsymmetric
        The trained model with the best LCS loaded.
    """

    # ----------------------------------------------------------------
    # 1. Sort cells and check pseudotime reversal status
    # ----------------------------------------------------------------
    if not np.all(np.diff(adata.obs['pseudotime'].values) >= 0):
        adata = adata[adata.obs['pseudotime'].argsort()].copy()

    pt_reversed = adata.uns.get('pseudotime_reversed', False)

    # 2. Normalise RNA (log1p)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ----------------------------------------------------------------
    # 3. Latent projections (full dataset)
    # ----------------------------------------------------------------
    n_components = 50
    # RNA PCA
    pca = PCA(n_components=n_components, random_state=42)
    rna_pca = pca.fit_transform(
        adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    )
     # ATAC TF‑IDF + Truncated SVD
    atac_mat = adata.obsm['ATAC_gene']
    if not hasattr(atac_mat, 'toarray'):
        atac_mat = atac_mat.toarray()
    tfidf = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True)
    atac_tfidf = tfidf.fit_transform(atac_mat)
    lsi = TruncatedSVD(n_components=n_components, random_state=42)
    atac_lsi = lsi.fit_transform(atac_tfidf)

    # ----------------------------------------------------------------
    # 4. Save transformers for reuse by validation scripts
    # ----------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(output_path / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(output_path / "lsi.pkl", "wb") as f:
        pickle.dump(lsi, f)

    # ----------------------------------------------------------------
    # 5. CCF target (full dataset)
    # ----------------------------------------------------------------
    pt = adata.obs['pseudotime'].values
    ccf_target = _compute_ccf_target(atac_lsi, rna_pca, pt, pt_reversed)
    ccf_target_t = torch.tensor(ccf_target, dtype=torch.float32)
    valid_mask = ~torch.isnan(ccf_target_t)

    # 6. Convert latent representations to tensors
    rna = torch.tensor(rna_pca, dtype=torch.float32)
    atac = torch.tensor(atac_lsi, dtype=torch.float32)
    pt_t = torch.tensor(pt, dtype=torch.float32)

    # ----------------------------------------------------------------
    # 7. Build model, optimiser, scheduler
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 8. Training loop
    # ----------------------------------------------------------------
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

            # Reconstruction loss (Pass 2 output)
            loss_recon = mse_loss(out['rna_pred'], rna_b)

            # Lag consistency loss: per‑component mean lag vs CCF target
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

        # ----------------------------------------------------------------
        # 9. In‑set LCS evaluation and checkpointing
        # ----------------------------------------------------------------
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

    # Load best model
    model.load_state_dict(best_state)
    if save_model and best_state is not None:
        torch.save(best_state, save_model)
    print(f"Training finished. Best LCS: {best_lcs:.4f}")

    # ----------------------------------------------------------------
    # 10. Save training metrics as CSV
    # ----------------------------------------------------------------
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
    """Train the symmetric baseline (no attention bias, no lag loss).

    The architecture, data preprocessing, and optimisation are identical
    to ``train_asymmetric``.  The only differences are:

        - The model is called with ``use_bias=False``, so cross‑attention
          is applied symmetrically without any temporal bias.
        - The lag consistency loss is **not** included in the training
          objective; the model minimises only the reconstruction loss.

    The CCF target is still computed and LCS is reported, but the model
    has no incentive to learn the temporal ordering of components,
    providing a clear contrast to the asymmetric model.

    Parameters
    ----------
    adata : AnnData
        Same requirements as in ``train_asymmetric``.
    epochs : int, default=50
        Total number of training epochs.
    val_interval : int, default=5
        Compute LCS every N epochs.
    batch_size : int, default=32
        Mini‑batch size.
    lr : float, default=1e-3
        Learning rate.
    device : str, default='cpu'
        Device for training.
    save_model : str, optional
        Path to save the best model checkpoint.
    output_dir : str, default='./tmo_results'
        Directory for metrics CSV and pickled transformers.

    Returns
    -------
    TMOLatentModelAsymmetric
        The trained symmetric baseline model.
    """

    # ----------------------------------------------------------------
    # 1. Sort cells and check pseudotime reversal status
    # ----------------------------------------------------------------
    if not np.all(np.diff(adata.obs['pseudotime'].values) >= 0):
        adata = adata[adata.obs['pseudotime'].argsort()].copy()

    pt_reversed = adata.uns.get('pseudotime_reversed', False)

    # 2. Normalise RNA
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # ----------------------------------------------------------------
    # 3. Latent projections (full dataset)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 4. Save transformers
    # ----------------------------------------------------------------
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(output_path / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(output_path / "lsi.pkl", "wb") as f:
        pickle.dump(lsi, f)

    # ----------------------------------------------------------------
    # 5. CCF target (computed but not used in loss)
    # ----------------------------------------------------------------
    pt = adata.obs['pseudotime'].values
    # Still compute CCF target so that LCS can be reported (but it is not used in loss)
    ccf_target = _compute_ccf_target(atac_lsi, rna_pca, pt, pt_reversed)
    ccf_target_t = torch.tensor(ccf_target, dtype=torch.float32)

    # 6. Tensors
    rna = torch.tensor(rna_pca, dtype=torch.float32)
    atac = torch.tensor(atac_lsi, dtype=torch.float32)
    pt_t = torch.tensor(pt, dtype=torch.float32)

    # ----------------------------------------------------------------
    # 7. Model (identical architecture)
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 8. Training loop (reconstruction only)
    # ----------------------------------------------------------------
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

        # ----------------------------------------------------------------
        # 9. In‑set LCS evaluation (purely diagnostic)
        # ----------------------------------------------------------------
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

    # Load best model
    model.load_state_dict(best_state)
    if save_model and best_state is not None:
        torch.save(best_state, save_model)
    print(f"Training finished. Best LCS: {best_lcs:.4f}")

    # ----------------------------------------------------------------
    # 10. Save metrics CSV
    # ----------------------------------------------------------------
    df = pd.DataFrame(lcs_history, columns=["epoch", "lcs", "loss"])
    df.to_csv(output_path / "training_metrics_tmo_symmetric.csv", index=False)
    return model