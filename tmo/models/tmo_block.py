"""TMO transformer block and encoder stack.

This module defines the core computational unit of the TMO model:
the ``TMOBlock``, which processes ATAC and RNA token sequences through
three sub‑stages:

  1. **Within‑modality self‑attention** – standard multi‑head self‑attention
     applied independently to the ATAC and RNA token sequences.
  2. **Asymmetric cross‑modal attention** – cross‑attention from ATAC to RNA
     and from RNA to ATAC, with an optional lag‑dependent bias that encodes
     the expected temporal offset between chromatin and transcription.
  3. **Position‑wise feed‑forward network** – a two‑layer MLP with GELU
     activation, shared across positions but with separate parameters for
     ATAC and RNA streams.

Each sub‑stage is wrapped in a residual connection and preceded by LayerNorm.

The ``TMOEncoder`` class simply stacks ``num_layers`` of ``TMOBlock`` modules
to form the complete encoder.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import WithinModalitySelfAttention, CrossAttentionPair


class TMOBlock(nn.Module):
    """A single TMO transformer layer.

    This layer processes ATAC and RNA token sequences in parallel,
    combining self‑attention within each modality, asymmetric cross‑attention
    between modalities, and a shared feed‑forward network.

    Parameters
    ----------
    d_model : int
        Dimensionality of the token embeddings.
    n_heads : int
        Number of attention heads for both self‑ and cross‑attention.
    dropout : float, default=0.1
        Dropout probability applied after each activation.
    expansion_factor : int, default=4
        Multiplier for the hidden dimension of the feed‑forward network.
    use_windowed_sparsity : bool, default=False
        If True, use windowed sparse attention in the cross‑attention modules.
    bucket_size : int, default=50
        Bucket size for windowed sparse attention (only relevant when
        ``use_windowed_sparsity`` is True).
    delta_tau_max : float, default=0.5
        Maximum absolute lag (pseudotime units).  Used to define the
        attention window when sparsity is enabled.
    sigma_max : float, default=0.2
        Maximum expected window width (standard deviation).  Used together
        with ``delta_tau_max`` to determine the sparse attention radius.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        use_windowed_sparsity: bool = False,
        bucket_size: int = 50,
        delta_tau_max: float = 0.5,
        sigma_max: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Self‑attention for each modality (no temporal bias).
        self.self_atac = WithinModalitySelfAttention(d_model, n_heads, dropout)
        self.self_rna = WithinModalitySelfAttention(d_model, n_heads, dropout)

        # Bidirectional cross‑attention with lag‑dependent bias.
        self.cross_attn = CrossAttentionPair(
            d_model, n_heads, dropout,
            use_windowed_sparsity=use_windowed_sparsity,
            bucket_size=bucket_size,
            delta_tau_max=delta_tau_max,
            sigma_max=sigma_max,
        )

        # Layer norms after each sub‑stage.
        self.norm_atac1 = nn.LayerNorm(d_model)
        self.norm_rna1 = nn.LayerNorm(d_model)
        self.norm_atac2 = nn.LayerNorm(d_model)
        self.norm_rna2 = nn.LayerNorm(d_model)

        # Shared feed‑forward network (applied independently to each token).
        self.ffn = nn.Sequential(
            nn.Linear(d_model, expansion_factor * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion_factor * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm_atac_ffn = nn.LayerNorm(d_model)
        self.norm_rna_ffn = nn.LayerNorm(d_model)

    def forward(
        self,
        atac_tokens: torch.Tensor,
        rna_tokens: torch.Tensor,
        tau_atac: torch.Tensor,
        tau_rna: torch.Tensor,
        delta_tau_atac_to_rna: Optional[torch.Tensor] = None,
        sigma_sq_atac_to_rna: Optional[torch.Tensor] = None,
        delta_tau_rna_to_atac: Optional[torch.Tensor] = None,
        sigma_sq_rna_to_atac: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of a single TMO block.

        Parameters
        ----------
        atac_tokens : torch.Tensor, shape (batch, seq_len_atac, d_model)
        rna_tokens : torch.Tensor, shape (batch, seq_len_rna, d_model)
        tau_atac : torch.Tensor, shape (batch, seq_len_atac)
            Pseudotime values for each ATAC token (same for all cells in
            the batch, repeated along the sequence).
        tau_rna : torch.Tensor, shape (batch, seq_len_rna)
            Pseudotime values for each RNA token.
        delta_tau_atac_to_rna : torch.Tensor or None, shape broadcastable to (batch, seq_len_atac, seq_len_rna)
            Predicted lags for the ATAC→RNA direction.  If None, no bias
            is applied (unbiased cross‑attention).
        sigma_sq_atac_to_rna : torch.Tensor or None
            Squared window widths for the ATAC→RNA bias.
        delta_tau_rna_to_atac : torch.Tensor or None
            Predicted lags for the RNA→ATAC direction.
        sigma_sq_rna_to_atac : torch.Tensor or None
            Squared window widths for the RNA→ATAC bias.
        attn_mask : torch.Tensor, optional
            Additional attention mask (e.g., padding mask).

        Returns
        -------
        atac_tokens : torch.Tensor, shape (batch, seq_len_atac, d_model)
            Updated ATAC token representations.
        rna_tokens : torch.Tensor, shape (batch, seq_len_rna, d_model)
            Updated RNA token representations.
        """

        # 1. Within‑modality self‑attention (residual + LayerNorm)
        atac_self = self.self_atac(atac_tokens, attn_mask)
        atac_tokens = self.norm_atac1(atac_tokens + atac_self)
        rna_self = self.self_rna(rna_tokens, attn_mask)
        rna_tokens = self.norm_rna1(rna_tokens + rna_self)

        # 2. Asymmetric cross‑modal attention (residual + LayerNorm)
        atac_cross, rna_cross = self.cross_attn(
            atac_tokens, rna_tokens,
            tau_atac, tau_rna,
            delta_tau_atac_to_rna, sigma_sq_atac_to_rna,
            delta_tau_rna_to_atac, sigma_sq_rna_to_atac,
            attn_mask,
        )
        atac_tokens = self.norm_atac2(atac_tokens + atac_cross)
        rna_tokens = self.norm_rna2(rna_tokens + rna_cross)

        # 3. Feed‑forward network (residual + LayerNorm)
        atac_ffn = self.ffn(atac_tokens)
        atac_tokens = self.norm_atac_ffn(atac_tokens + atac_ffn)
        rna_ffn = self.ffn(rna_tokens)
        rna_tokens = self.norm_rna_ffn(rna_tokens + rna_ffn)

        return atac_tokens, rna_tokens


class TMOEncoder(nn.Module):
    """Stack of ``TMOBlock`` layers forming the TMO encoder.

    The encoder sequentially applies ``num_layers`` identical TMO blocks.
    Each block receives the ATAC and RNA token sequences together with the
    pseudotime values and, when available, the predicted lags and widths
    that bias the cross‑attention.

    Parameters
    ----------
    num_layers : int
        Number of TMO transformer blocks.
    d_model : int
        Token embedding dimension.
    n_heads : int
        Number of attention heads per block.
    dropout : float, default=0.1
        Dropout probability.
    expansion_factor : int, default=4
        FFN hidden dimension multiplier.
    use_windowed_sparsity : bool, default=False
        Enable windowed sparse attention (passed to each block).
    bucket_size : int, default=50
        Bucket size for sparse attention.
    delta_tau_max : float, default=0.5
        Maximum absolute lag for sparse attention.
    sigma_max : float, default=0.2
        Maximum window width for sparse attention.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        expansion_factor: int = 4,
        use_windowed_sparsity: bool = False,
        bucket_size: int = 50,
        delta_tau_max: float = 0.5,
        sigma_max: float = 0.2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TMOBlock(
                d_model, n_heads, dropout, expansion_factor,
                use_windowed_sparsity, bucket_size, delta_tau_max, sigma_max,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        atac_tokens: torch.Tensor,
        rna_tokens: torch.Tensor,
        tau_atac: torch.Tensor,
        tau_rna: torch.Tensor,
        delta_tau_atac_to_rna: Optional[torch.Tensor] = None,
        sigma_sq_atac_to_rna: Optional[torch.Tensor] = None,
        delta_tau_rna_to_atac: Optional[torch.Tensor] = None,
        sigma_sq_rna_to_atac: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass token sequences through the encoder stack.

        Parameters
        ----------
        atac_tokens, rna_tokens : torch.Tensor
            Token sequences of shape (batch, seq_len, d_model).
        tau_atac, tau_rna : torch.Tensor
            Pseudotime values for each position.
        delta_tau_atac_to_rna, sigma_sq_atac_to_rna : optional
            Lag and width predictions for the forward bias.
        delta_tau_rna_to_atac, sigma_sq_rna_to_atac : optional
            Lag and width predictions for the reverse bias.
        attn_mask : optional
            Additional attention mask.

        Returns
        -------
        atac_tokens, rna_tokens : torch.Tensor
            Output token sequences after ``num_layers`` transformer blocks.
        """
        
        for layer in self.layers:
            atac_tokens, rna_tokens = layer(
                atac_tokens, rna_tokens,
                tau_atac, tau_rna,
                delta_tau_atac_to_rna, sigma_sq_atac_to_rna,
                delta_tau_rna_to_atac, sigma_sq_rna_to_atac,
                attn_mask,
            )
        return atac_tokens, rna_tokens