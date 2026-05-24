"""Full TMO model: tokenizer, encoder, decoder, and lag predictors.

This module defines the latent-space asymmetric TMO model used in the publication.
It operates on pre‑computed PCA (RNA) and LSI (ATAC) components, tokenising each
cell as a sequence of 100 tokens: 50 RNA components + 50 ATAC components.

Token embeddings are formed by projecting the scalar latent value through
a linear layer and adding modality and pseudotime embeddings.  This design
ensures every token carries cell‑state‑specific information while also
encoding its modality and the cell's pseudotime coordinate. Token embeddings are:
    x = component_embedding(comp_idx) + modality_embedding(mod) + pseudotime_embedding(τ)

The model performs two forward passes:
    1. **First pass (unbiased)** — tokens are processed by the transformer
     encoder with all cross‑attention biases set to zero.  The ATAC token
     outputs are mean‑pooled to obtain a *cell embedding*, z_c, which
     summarises the cell's chromatin state.

    2. **Lag and width prediction** — for every component (ATAC and RNA),
     a LagMLP and WidthMLP take as input the concatenation of z_c and the
     component's learned embedding.  They predict a signed regulatory lag
     Δτ̂ and a window width σ̂².

    3. **Second pass (biased)** — the same tokens are fed through the encoder
     again, but now cross‑attention is biased by the predicted lags and
     widths.  An ATAC token attending to an RNA token receives an additive
     Gaussian bias centred at the expected temporal offset.  The RNA token
     outputs are decoded to reconstruct the true RNA latent components.

The `use_bias` flag controls whether the second pass is executed.  When set
to `False` (symmetric baseline), only the first pass runs and no lag loss
is applied during training.  This allows a fair comparison between the
asymmetric model and a structurally identical but temporally agnostic
baseline. 
    
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
        Dimensionality of all token embeddings.
    n_heads : int, default=4
        Number of attention heads in every self‑ and cross‑attention layer.
    num_layers : int, default=2
        Number of sequential TMO transformer blocks.
    dropout : float, default=0.1
        Dropout probability applied after every GELU activation.
    expansion_factor : int, default=4
        Multiplier for the hidden dimension of the feed‑forward network
        inside each transformer block.
    use_windowed_sparsity : bool, default=False
        If True, cross‑attention is restricted to token pairs whose
        pseudotime difference is within a few standard deviations of the
        predicted lag.  This is used for scalability on large datasets.
    delta_tau_max : float, default=0.5
        Maximum absolute value of the predicted regulatory lag (pseudotime
        units).  The LagMLP output is clipped to [-Δτ_max, Δτ_max].
    sigma_sq_min : float, default=0.001
        Minimum value for the predicted squared window width.  Guarantees
        numerical stability when the width appears in the denominator of
        the attention bias.
    lag_hidden_dims : list of int, optional
        Hidden layer dimensions for both the LagMLP and the WidthMLP.
        Defaults to [d_model, d_model // 2].
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

        # ------------------------------------------------------------------
        # Tokenizer: converts latent vectors to token embeddings.
        # The forward signature is tokenizer(rna_latent, atac_latent, pseudotime).
        # Internally it applies a learned linear projection to each scalar
        # value and adds modality and pseudotime embeddings. Therefore, it is
        # component‑index based, no latent values.
        # ------------------------------------------------------------------
        self.tokenizer = LatentTokenizer(
            n_rna_components=n_rna_components,
            n_atac_components=n_atac_components,
            d_model=d_model,
            pseudotime_hidden_dims=[128, 64],
        )

        # ------------------------------------------------------------------
        # Transformer encoder: stack of TMOBlock layers.
        # Each block contains within‑modality self‑attention, asymmetric
        # cross‑attention, and a position‑wise feed‑forward network.
        # ------------------------------------------------------------------
        self.encoder = TMOEncoder(
            num_layers=num_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            use_windowed_sparsity=use_windowed_sparsity,
            delta_tau_max=delta_tau_max,
        )

        # ------------------------------------------------------------------
        # Decoder: projects each RNA token output to a scalar value.
        # After the second pass, the output of every RNA token is linearly
        # mapped to predict the corresponding RNA latent component.
        # ------------------------------------------------------------------
        self.rna_decoder = nn.Linear(d_model, 1)

        # ------------------------------------------------------------------
        # Component embeddings: learned vectors that identify each latent
        # component.  They are used by the LagMLP / WidthMLP, together with
        # the cell embedding, to produce component‑specific lag predictions.
        # These are separate from any embeddings used in the tokenizer.
        # ------------------------------------------------------------------
        self.atac_component_emb = nn.Embedding(n_atac_components, d_model)
        self.rna_component_emb = nn.Embedding(n_rna_components, d_model)

        # ------------------------------------------------------------------
        # Lag and width predictor: a combined module containing both a
        # LagMLP and a WidthMLP.  It is applied once per (cell, component)
        # pair.  The `detach_cell_embedding` flag is set to False in the
        # final pipeline, so gradients flow from the lag loss back into
        # the encoder, enabling end‑to‑end learning of the cell state.
        # ------------------------------------------------------------------
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
        """Execute the two‑pass (or single‑pass) forward of the TMO model.

        Parameters
        ----------
        rna_latent : torch.Tensor, shape (batch, n_rna)
            Pre‑computed RNA PCA components for each cell.
        atac_latent : torch.Tensor, shape (batch, n_atac)
            Pre‑computed ATAC LSI components for each cell.
        pseudotime : torch.Tensor, shape (batch,)
            Pseudotime value for each cell, in [0, 1].
        use_bias : bool, default=True
            If True, run the full two‑pass asymmetric mechanism.  If False,
            only the first (unbiased) pass is executed and the predicted
            lags are returned without influencing the encoder — this
            serves as the symmetric baseline.

        Returns
        -------
        dict
            'rna_pred' : torch.Tensor, shape (batch, n_rna)
                Reconstructed RNA latent components.  When ``use_bias`` is
                True this comes from the second (biased) pass; otherwise
                it is the first‑pass reconstruction.
            'rna_pred_pass1' : torch.Tensor, shape (batch, n_rna)
                Reconstruction from the first pass (always provided).
            'lag_per_token' : torch.Tensor, shape (batch, n_atac + n_rna)
                Predicted regulatory lags for every token (ATAC then RNA).
            'sigma_sq' : torch.Tensor, shape (batch, n_atac + n_rna)
                Predicted window widths squared for every token.
            'cell_emb' : torch.Tensor, shape (batch, d_model)
                Cell embedding obtained by mean‑pooling the ATAC token
                outputs of the first pass.
        """

        batch = rna_latent.shape[0]
        n_atac = self.n_atac
        n_rna = self.n_rna

        # ================================================================
        # Pass 1 – unbiased encoder pass
        # ================================================================
        # Tokenise the latent vectors.
        tokens1 = self.tokenizer(rna_latent, atac_latent, pseudotime)
        # Pseudotime values are expanded to match the full sequence length.
        tau = pseudotime.unsqueeze(-1).expand(-1, n_atac + n_rna)

        atac_tok1 = tokens1[:, :n_atac, :] # first n_atac tokens
        rna_tok1 = tokens1[:, n_atac:, :] # remaining n_rna tokens

        # Forward through the encoder with no cross‑attention bias.
        atac_enc1, rna_enc1 = self.encoder(
            atac_tok1, rna_tok1,
            tau_atac=tau[:, :n_atac],
            tau_rna=tau[:, n_atac:],
            delta_tau_atac_to_rna=None,
            sigma_sq_atac_to_rna=None,
            delta_tau_rna_to_atac=None,
            sigma_sq_rna_to_atac=None)

        # Cell embedding: mean over the ATAC token axis.
        cell_emb = atac_enc1.mean(dim=1) # not detached, (batch, d_model)
        if getattr(self, 'ablate_cell_state', False):
            cell_emb = torch.zeros_like(cell_emb)
        # RNA reconstruction from the first pass (used for monitoring).
        rna_pred_pass1 = self.rna_decoder(rna_enc1).squeeze(-1)

        # ================================================================
        # Lag and width prediction
        # ================================================================
        # For each ATAC component, predict Δτ^ and σ̂².
        atac_comp_ids = torch.arange(n_atac, device=rna_latent.device).unsqueeze(0).expand(batch, -1)
        atac_comp_emb = self.atac_component_emb(atac_comp_ids)
        delta_tau_atac, sigma_sq_atac = self.lag_predictor(
            cell_emb.unsqueeze(1).expand(-1, n_atac, -1).reshape(-1, self.d_model),
            atac_comp_emb.reshape(-1, self.d_model))
        delta_tau_atac = delta_tau_atac.reshape(batch, n_atac)
        sigma_sq_atac = sigma_sq_atac.reshape(batch, n_atac)

        # For each RNA component, predict Δτ^ and σ̂².
        rna_comp_ids = torch.arange(n_rna, device=rna_latent.device).unsqueeze(0).expand(batch, -1)
        rna_comp_emb = self.rna_component_emb(rna_comp_ids)
        delta_tau_rna, sigma_sq_rna = self.lag_predictor(
            cell_emb.unsqueeze(1).expand(-1, n_rna, -1).reshape(-1, self.d_model),
            rna_comp_emb.reshape(-1, self.d_model))
        delta_tau_rna = delta_tau_rna.reshape(batch, n_rna)
        sigma_sq_rna = sigma_sq_rna.reshape(batch, n_rna)

        # Concatenate predictions for output.
        lag_per_token = torch.cat([delta_tau_atac, delta_tau_rna], dim=1)
        sigma_sq_all = torch.cat([sigma_sq_atac, sigma_sq_rna], dim=1)

        # ================================================================
        # Symmetric baseline – return without biased pass
        # ================================================================
        if not use_bias:
            # symmetric baseline – return only pass‑1 reconstruction
            return {
                'rna_pred': rna_pred_pass1, # same as pass‑1 recon
                'rna_pred_pass1': rna_pred_pass1,
                'lag_per_token': lag_per_token,
                'sigma_sq': sigma_sq_all,
                'cell_emb': cell_emb,
            }

        # ================================================================
        # Pass 2 – biased encoder pass
        # ================================================================
        # Prepare the bias matrices.  For ATAC->RNA the bias uses the
        # RNA‑component lags; for RNA→ATAC it uses the ATAC‑component lags.
        delta_tau_atac_to_rna = delta_tau_rna.unsqueeze(1)   # (batch, 1, n_rna)
        sigma_sq_atac_to_rna = sigma_sq_rna.unsqueeze(1)
        delta_tau_rna_to_atac = delta_tau_atac.unsqueeze(1)  # (batch, 1, n_atac)
        sigma_sq_rna_to_atac = sigma_sq_atac.unsqueeze(1)

        # Tokenise again (can reuse; repeated for clarity).
        tokens2 = self.tokenizer(rna_latent, atac_latent, pseudotime)
        atac_tok2 = tokens2[:, :n_atac, :]
        rna_tok2 = tokens2[:, n_atac:, :]

        # Encoder with asymmetric bias.
        atac_enc2, rna_enc2 = self.encoder(
            atac_tok2, rna_tok2,
            tau_atac=tau[:, :n_atac], tau_rna=tau[:, n_atac:],
            delta_tau_atac_to_rna=delta_tau_atac_to_rna,
            sigma_sq_atac_to_rna=sigma_sq_atac_to_rna,
            delta_tau_rna_to_atac=delta_tau_rna_to_atac,
            sigma_sq_rna_to_atac=sigma_sq_rna_to_atac)

        # Decode RNA from the biased representations.
        rna_pred = self.rna_decoder(rna_enc2).squeeze(-1)

        return {
            'rna_pred': rna_pred,
            'rna_pred_pass1': rna_pred_pass1,
            'lag_per_token': lag_per_token,
            'sigma_sq': sigma_sq_all,
            'cell_emb': cell_emb,
        }