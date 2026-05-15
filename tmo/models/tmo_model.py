"""Full TMO model: tokenizer, encoder, decoder, and lag predictors.

This module defines the latent-space asymmetric TMO model used in the publication.
It operates on pre‑computed PCA (RNA) and LSI (ATAC) components, tokenising each
cell as a sequence of 100 tokens: 50 RNA components + 50 ATAC components.

Token embeddings are:
    x = component_embedding(comp_idx) + modality_embedding(mod) + pseudotime_embedding(τ)

The model performs two forward passes:
    1. First pass (no bias) → cell embedding from ATAC tokens (mean pool).
    2. Predict Δτ and σ² per component from the detached cell embedding.
    3. Second pass (with asymmetric attention bias) → reconstruct RNA components.

Gradients from the reconstruction loss (Pass 2) do NOT flow back into the cell
embedding, keeping the conceptual separation between cell‑state and regulatory lag.
The encoder is trained via a lightweight reconstruction loss on the first pass.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .tokenizer import LatentTokenizer
from .tmo_block import TMOEncoder
from .lagmlp import CombinedLagWidthPredictor


class TMOLatentModelAsymmetric(nn.Module):
    """TMO latent model with asymmetric attention biased by predicted lags.

    Parameters
    ----------
    n_rna_components : int
        Number of RNA latent components (e.g., 50).
    n_atac_components : int
        Number of ATAC latent components (e.g., 50).
    d_model : int, default=64
        Embedding dimension for all tokens.
    n_heads : int, default=4
        Number of attention heads.
    num_layers : int, default=2
        Number of TMO transformer layers.
    dropout : float, default=0.1
        Dropout probability.
    expansion_factor : int, default=4
        Expansion factor for the feed‑forward network.
    use_windowed_sparsity : bool, default=False
        If True, use windowed sparse attention (for large datasets).
    delta_tau_max : float, default=0.5
        Maximum absolute regulatory lag (pseudotime units).
    sigma_sq_min : float, default=0.001
        Minimum value for the predicted window width squared.
    lag_hidden_dims : list of int, optional
        Hidden layer dimensions for LagMLP and WidthMLP.
    """

    def __init__(
        self,
        n_rna_components: int,
        n_atac_components: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        use_windowed_sparsity: bool = False,
        delta_tau_max: float = 0.5,
        sigma_sq_min: float = 0.001,
        lag_hidden_dims: Optional[list] = None,
    ):
        super().__init__()
        self.n_rna = n_rna_components
        self.n_atac = n_atac_components
        self.d_model = d_model
        self.delta_tau_max = delta_tau_max

        # Tokenizer (component‑index based, no latent values)
        self.tokenizer = LatentTokenizer(
            n_rna_components=n_rna_components,
            n_atac_components=n_atac_components,
            d_model=d_model,
            pseudotime_hidden_dims=[128, 64],
        )

        # Transformer encoder stack
        self.encoder = TMOEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            use_windowed_sparsity=use_windowed_sparsity,
            delta_tau_max=delta_tau_max,
        )

        # Decoder for RNA reconstruction (projects from token output to scalar)
        self.rna_decoder = nn.Linear(d_model, 1)

        # Component embeddings used by the lag predictor (separate from tokenizer)
        # These are the e_comp(i) vectors that are combined with the cell embedding.
        self.atac_component_emb = nn.Embedding(n_atac_components, d_model)
        self.rna_component_emb = nn.Embedding(n_rna_components, d_model)

        # Lag and width predictor (applied per component per cell)
        self.lag_predictor = CombinedLagWidthPredictor(
            d_embed=d_model,
            lag_hidden_dims=lag_hidden_dims,
            width_hidden_dims=lag_hidden_dims,
            delta_tau_max=delta_tau_max,
            sigma_sq_min=sigma_sq_min,
            dropout=dropout,
            detach_cell_embedding=False,   # we manually un-detach to be explicit
        )

    def forward(self, rna_latent, atac_latent, pseudotime, use_bias=True):
        batch = rna_latent.shape[0]
        n_atac = self.n_atac
        n_rna = self.n_rna

        # Pass 1 (unbiased)
        tokens1 = self.tokenizer(rna_latent, atac_latent, pseudotime)
        tau = pseudotime.unsqueeze(-1).expand(-1, n_atac + n_rna)
        atac_tok1 = tokens1[:, :n_atac, :]
        rna_tok1 = tokens1[:, n_atac:, :]

        atac_enc1, rna_enc1 = self.encoder(
            atac_tok1, rna_tok1,
            tau_atac=tau[:, :n_atac], tau_rna=tau[:, n_atac:],
            delta_tau_atac_to_rna=None, sigma_sq_atac_to_rna=None,
            delta_tau_rna_to_atac=None, sigma_sq_rna_to_atac=None)

        cell_emb = atac_enc1.mean(dim=1)                 # not detached
        rna_pred_pass1 = self.rna_decoder(rna_enc1).squeeze(-1)

        # Predict lags and widths
        atac_comp_ids = torch.arange(n_atac, device=rna_latent.device).unsqueeze(0).expand(batch, -1)
        atac_comp_emb = self.atac_component_emb(atac_comp_ids)
        delta_tau_atac, sigma_sq_atac = self.lag_predictor(
            cell_emb.unsqueeze(1).expand(-1, n_atac, -1).reshape(-1, self.d_model),
            atac_comp_emb.reshape(-1, self.d_model))
        delta_tau_atac = delta_tau_atac.reshape(batch, n_atac)
        sigma_sq_atac = sigma_sq_atac.reshape(batch, n_atac)

        rna_comp_ids = torch.arange(n_rna, device=rna_latent.device).unsqueeze(0).expand(batch, -1)
        rna_comp_emb = self.rna_component_emb(rna_comp_ids)
        delta_tau_rna, sigma_sq_rna = self.lag_predictor(
            cell_emb.unsqueeze(1).expand(-1, n_rna, -1).reshape(-1, self.d_model),
            rna_comp_emb.reshape(-1, self.d_model))
        delta_tau_rna = delta_tau_rna.reshape(batch, n_rna)
        sigma_sq_rna = sigma_sq_rna.reshape(batch, n_rna)

        lag_per_token = torch.cat([delta_tau_atac, delta_tau_rna], dim=1)
        sigma_sq_all = torch.cat([sigma_sq_atac, sigma_sq_rna], dim=1)

        if not use_bias:
            # symmetric baseline – return only pass‑1 reconstruction
            return {
                'rna_pred': rna_pred_pass1,
                'rna_pred_pass1': rna_pred_pass1,
                'lag_per_token': lag_per_token,
                'sigma_sq': sigma_sq_all,
                'cell_emb': cell_emb,
            }

        # Pass 2 with bias
        delta_tau_atac_to_rna = delta_tau_rna.unsqueeze(1)   # (batch,1,n_rna)
        sigma_sq_atac_to_rna = sigma_sq_rna.unsqueeze(1)
        delta_tau_rna_to_atac = delta_tau_atac.unsqueeze(1)  # (batch,1,n_atac)
        sigma_sq_rna_to_atac = sigma_sq_atac.unsqueeze(1)

        tokens2 = self.tokenizer(rna_latent, atac_latent, pseudotime)
        atac_tok2 = tokens2[:, :n_atac, :]
        rna_tok2 = tokens2[:, n_atac:, :]

        atac_enc2, rna_enc2 = self.encoder(
            atac_tok2, rna_tok2,
            tau_atac=tau[:, :n_atac], tau_rna=tau[:, n_atac:],
            delta_tau_atac_to_rna=delta_tau_atac_to_rna,
            sigma_sq_atac_to_rna=sigma_sq_atac_to_rna,
            delta_tau_rna_to_atac=delta_tau_rna_to_atac,
            sigma_sq_rna_to_atac=sigma_sq_rna_to_atac)

        rna_pred = self.rna_decoder(rna_enc2).squeeze(-1)

        return {
            'rna_pred': rna_pred,
            'rna_pred_pass1': rna_pred_pass1,
            'lag_per_token': lag_per_token,
            'sigma_sq': sigma_sq_all,
            'cell_emb': cell_emb,
        }