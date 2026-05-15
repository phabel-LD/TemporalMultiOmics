"""Loss functions for TMO training.

Includes:
    - L_recon: MSE between predicted and observed RNA expression.
    - L_align: InfoNCE contrastive loss aligning ATAC and RNA cell embeddings.
    - L_lag: Lag consistency loss (MSE between learned Δτ̂ and CCF prior).
    - L_pert: MSE for perturbation prediction (Perturb-seq).
    - L_lagmlp: Supervised pretraining loss for LagMLP (weighted MSE against CCF surface).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean squared error loss with optional masking.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Target values (same shape as pred).
    mask : torch.Tensor, optional
        Binary mask (1 for valid positions, 0 for ignore). Same shape as pred.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    if mask is not None:
        diff = (pred - target) ** 2
        diff = diff * mask
        loss = diff.sum() / (mask.sum() + 1e-8)
    else:
        loss = F.mse_loss(pred, target)
    return loss


def infonce_loss(
    z_atac: torch.Tensor,
    z_rna: torch.Tensor,
    temperature: float = 0.07,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """InfoNCE contrastive loss for aligning ATAC and RNA cell embeddings.

    For each cell c, we want the embedding from ATAC (z_atac[c]) to be closer to
    its paired RNA embedding (z_rna[c]) than to RNA embeddings of other cells.

    Parameters
    ----------
    z_atac : torch.Tensor, shape (batch, d_model)
        Pooled ATAC embeddings.
    z_rna : torch.Tensor, shape (batch, d_model)
        Pooled RNA embeddings.
    temperature : float, default=0.07
        Temperature scaling factor.
    mask : torch.Tensor, optional
        Binary mask for valid cells (e.g., exclude low-quality cells). Shape (batch,).

    Returns
    -------
    torch.Tensor
        Scalar InfoNCE loss.
    """
    batch = z_atac.shape[0]
    # Normalize embeddings
    z_atac = F.normalize(z_atac, dim=1)
    z_rna = F.normalize(z_rna, dim=1)

    # Compute similarity matrix: (batch, batch)
    sim = z_atac @ z_rna.T  # cosine similarity because normalized
    # Scale by temperature
    sim = sim / temperature

    # Positive pairs: diagonal
    labels = torch.arange(batch, device=z_atac.device)

    # Apply mask if provided (set invalid rows to -inf so they don't contribute)
    if mask is not None:
        # Mask shape (batch,)
        mask = mask.bool()
        # For rows where mask is False, set sim row to -inf (so they don't affect loss)
        sim = sim.masked_fill(~mask.unsqueeze(1), float('-inf'))

    loss = F.cross_entropy(sim, labels)
    return loss


def lag_consistency_loss(
    delta_tau_learned: torch.Tensor,
    delta_tau_prior: torch.Tensor,
    prior_variance: float = 0.01,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lag consistency loss L_lag.

    Penalizes deviation of learned Δτ̂_g from CCF prior Δτ_g^CCF.
    L_lag = (1/G) Σ_g (Δτ̂_g - Δτ_g^CCF)² / σ₀²

    Parameters
    ----------
    delta_tau_learned : torch.Tensor, shape (n_genes,)
        Learned lag parameters per gene.
    delta_tau_prior : torch.Tensor, shape (n_genes,)
        CCF-estimated prior lags.
    prior_variance : float, default=0.01
        σ₀², prior variance controlling regularization strength.
    mask : torch.Tensor, optional, shape (n_genes,)
        Mask for genes with valid prior (e.g., non-NaN).

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    diff = (delta_tau_learned - delta_tau_prior) ** 2
    if mask is not None:
        diff = diff * mask
        loss = diff.sum() / (mask.sum() * prior_variance + 1e-8)
    else:
        loss = diff.mean() / prior_variance
    return loss


def perturbation_prediction_loss(
    pred_expr: torch.Tensor,
    true_expr: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perturbation loss L_pert for predicting post-perturbation expression.

    Parameters
    ----------
    pred_expr : torch.Tensor
        Predicted RNA expression after perturbation.
    true_expr : torch.Tensor
        Observed RNA expression.
    mask : torch.Tensor, optional
        Mask for valid genes/cells.

    Returns
    -------
    torch.Tensor
        MSE loss.
    """
    return mse_loss(pred_expr, true_expr, mask)


def lagmlp_supervised_loss(
    delta_tau_pred: torch.Tensor,
    delta_tau_target: torch.Tensor,
    inverse_var_weights: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Supervised pretraining loss for LagMLP (L_lagmlp).

    Weighted MSE between predicted Δτ̂_{g,c} and CCF-derived Δτ_g(τ_c),
    with weights w_{g,c} = 1 / (GP posterior variance at τ_c for gene g).

    Parameters
    ----------
    delta_tau_pred : torch.Tensor, shape (n_cells, n_genes) or (n_pairs,)
        Predicted lags.
    delta_tau_target : torch.Tensor, same shape
        CCF-derived target lags (may contain NaNs for sparse windows).
    inverse_var_weights : torch.Tensor, same shape, optional
        Inverse variance weights (1 / posterior variance). If None, uniform weights.
    mask : torch.Tensor, same shape, optional
        Binary mask for valid target entries (non-NaN).

    Returns
    -------
    torch.Tensor
        Weighted MSE loss.
    """
    # Compute squared error
    diff = (delta_tau_pred - delta_tau_target) ** 2

    # Apply mask (if provided) to ignore NaNs in target
    if mask is None and delta_tau_target.isnan().any():
        # Automatically create mask from non-NaN positions
        mask = ~delta_tau_target.isnan()
    if mask is not None:
        diff = diff * mask
        n_valid = mask.sum()
    else:
        n_valid = delta_tau_pred.numel()

    # Apply inverse variance weights
    if inverse_var_weights is not None:
        diff = diff * inverse_var_weights

    loss = diff.sum() / (n_valid + 1e-8)
    return loss


class TMOLosses(nn.Module):
    """Container for all TMO losses, with configurable weights."""

    def __init__(
        self,
        lambda_recon: float = 1.0,
        lambda_align: float = 0.1,
        lambda_lag: float = 1.0,
        lambda_pert: float = 1.0,
        prior_variance: float = 0.01,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_align = lambda_align
        self.lambda_lag = lambda_lag
        self.lambda_pert = lambda_pert
        self.prior_variance = prior_variance
        self.temperature = temperature

    def forward(
        self,
        rna_pred: Optional[torch.Tensor] = None,
        rna_target: Optional[torch.Tensor] = None,
        z_atac: Optional[torch.Tensor] = None,
        z_rna: Optional[torch.Tensor] = None,
        delta_tau_learned: Optional[torch.Tensor] = None,
        delta_tau_prior: Optional[torch.Tensor] = None,
        pert_pred: Optional[torch.Tensor] = None,
        pert_target: Optional[torch.Tensor] = None,
        recon_mask: Optional[torch.Tensor] = None,
        align_mask: Optional[torch.Tensor] = None,
        lag_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute total loss and individual components.

        Parameters
        ----------
        rna_pred, rna_target : tensors for reconstruction.
        z_atac, z_rna : cell embeddings for contrastive alignment.
        delta_tau_learned, delta_tau_prior : per-gene lags for L_lag.
        pert_pred, pert_target : for perturbation loss.
        recon_mask, align_mask, lag_mask : optional masks.

        Returns
        -------
        total_loss : torch.Tensor
        loss_dict : dict with individual loss terms.
        """
        loss_dict = {}
        total = 0.0

        if rna_pred is not None and rna_target is not None:
            loss_recon = mse_loss(rna_pred, rna_target, recon_mask)
            loss_dict['recon'] = loss_recon.item()
            total += self.lambda_recon * loss_recon

        if z_atac is not None and z_rna is not None:
            loss_align = infonce_loss(z_atac, z_rna, self.temperature, align_mask)
            loss_dict['align'] = loss_align.item()
            total += self.lambda_align * loss_align

        if delta_tau_learned is not None and delta_tau_prior is not None:
            loss_lag = lag_consistency_loss(
                delta_tau_learned, delta_tau_prior,
                self.prior_variance, lag_mask
            )
            loss_dict['lag'] = loss_lag.item()
            total += self.lambda_lag * loss_lag

        if pert_pred is not None and pert_target is not None:
            loss_pert = perturbation_prediction_loss(pert_pred, pert_target)
            loss_dict['pert'] = loss_pert.item()
            total += self.lambda_pert * loss_pert

        loss_dict['total'] = total.item()
        return total, loss_dict