"""TMO transformer block and encoder stack."""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import WithinModalitySelfAttention, CrossAttentionPair


class TMOBlock(nn.Module):
    """Single TMO transformer layer."""

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

        self.self_atac = WithinModalitySelfAttention(d_model, n_heads, dropout)
        self.self_rna = WithinModalitySelfAttention(d_model, n_heads, dropout)

        self.cross_attn = CrossAttentionPair(
            d_model, n_heads, dropout,
            use_windowed_sparsity=use_windowed_sparsity,
            bucket_size=bucket_size,
            delta_tau_max=delta_tau_max,
            sigma_max=sigma_max,
        )

        self.norm_atac1 = nn.LayerNorm(d_model)
        self.norm_rna1 = nn.LayerNorm(d_model)
        self.norm_atac2 = nn.LayerNorm(d_model)
        self.norm_rna2 = nn.LayerNorm(d_model)

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
        # Self-attention
        atac_self = self.self_atac(atac_tokens, attn_mask)
        atac_tokens = self.norm_atac1(atac_tokens + atac_self)
        rna_self = self.self_rna(rna_tokens, attn_mask)
        rna_tokens = self.norm_rna1(rna_tokens + rna_self)

        # Cross-attention
        atac_cross, rna_cross = self.cross_attn(
            atac_tokens, rna_tokens,
            tau_atac, tau_rna,
            delta_tau_atac_to_rna, sigma_sq_atac_to_rna,
            delta_tau_rna_to_atac, sigma_sq_rna_to_atac,
            attn_mask,
        )
        atac_tokens = self.norm_atac2(atac_tokens + atac_cross)
        rna_tokens = self.norm_rna2(rna_tokens + rna_cross)

        # FFN
        atac_ffn = self.ffn(atac_tokens)
        atac_tokens = self.norm_atac_ffn(atac_tokens + atac_ffn)
        rna_ffn = self.ffn(rna_tokens)
        rna_tokens = self.norm_rna_ffn(rna_tokens + rna_ffn)

        return atac_tokens, rna_tokens


class TMOEncoder(nn.Module):
    """Stack of TMO blocks."""

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
        for layer in self.layers:
            atac_tokens, rna_tokens = layer(
                atac_tokens, rna_tokens,
                tau_atac, tau_rna,
                delta_tau_atac_to_rna, sigma_sq_atac_to_rna,
                delta_tau_rna_to_atac, sigma_sq_rna_to_atac,
                attn_mask,
            )
        return atac_tokens, rna_tokens