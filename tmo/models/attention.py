"""Asymmetric cross-modal attention with lag bias.

This module implements:
    - AsymmetricCrossAttention: cross-attention from ATAC to RNA (or RNA to ATAC)
      with bias terms b_ij and b_ji_rev that encode the regulatory lag.
    - WithinModalitySelfAttention: standard self-attention (no bias).
    - Windowed attention helper for sparse computation (Algorithm 4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WithinModalitySelfAttention(nn.Module):
    """Standard self-attention for tokens within the same modality.

    No temporal bias is applied.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, d_model)
            Input tokens.
        mask : torch.Tensor, shape (batch, seq_len) or (batch, seq_len, seq_len), optional
            Padding or causal mask.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, d_model)
            Self-attention output.
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, self.d_model)
        out = self.proj(out)
        return out


class AsymmetricCrossAttention(nn.Module):
    """Asymmetric cross-attention between ATAC and RNA tokens.

    This module computes attention from a query modality (e.g., ATAC) to a
    key/value modality (e.g., RNA) with a bias term b_ij that depends on:
        - pseudotime difference τ_j - τ_i
        - learnable or predicted lag Δτ̂_{g,c}
        - window width σ²

    The bias is computed as:
        b_ij = - ( (τ_j - τ_i - Δτ̂)² / (2 σ²) )

    For the reverse direction (RNA -> ATAC), the bias uses b_ji_rev:
        b_ji_rev = - ( (τ_i - τ_j + Δτ̂)² / (2 σ²) )
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        max_lag: float = 0.5,
        use_windowed_sparsity: bool = False,
        bucket_size: int = 50,
        delta_tau_max: float = 0.5,
        sigma_max: float = 0.2,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.max_lag = max_lag
        self.use_windowed_sparsity = use_windowed_sparsity
        self.bucket_size = bucket_size
        self.delta_tau_max = delta_tau_max
        self.sigma_max = sigma_max

        # Linear projections for query, key, value
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _compute_bias(
        self,
        tau_q: torch.Tensor,           # (batch, seq_len_q)
        tau_kv: torch.Tensor,          # (batch, seq_len_kv)
        delta_tau: torch.Tensor,       # (batch, seq_len_q, seq_len_kv) or broadcastable
        sigma_sq: torch.Tensor,        # (batch, seq_len_q, seq_len_kv) or broadcastable
        reverse: bool = False,
    ) -> torch.Tensor:
        """Compute the lag bias tensor.

        Parameters
        ----------
        tau_q : torch.Tensor
            Pseudotimes of query positions.
        tau_kv : torch.Tensor
            Pseudotimes of key/value positions.
        delta_tau : torch.Tensor
            Expected regulatory lag (positive = query leads key).
        sigma_sq : torch.Tensor
            Squared window width.
        reverse : bool, default=False
            If True, use reverse bias formula (b_ji_rev).

        Returns
        -------
        torch.Tensor, shape (batch, n_heads, seq_len_q, seq_len_kv)
            Bias to be added to attention logits (before softmax).
        """
        # Compute difference: tau_kv - tau_q
        # Broadcasting: (batch, seq_len_q, 1) vs (batch, 1, seq_len_kv)
        tau_q_exp = tau_q.unsqueeze(-1)   # (batch, seq_len_q, 1)
        tau_kv_exp = tau_kv.unsqueeze(1)  # (batch, 1, seq_len_kv)
        diff = tau_kv_exp - tau_q_exp     # (batch, seq_len_q, seq_len_kv)

        if reverse:
            # Reverse bias (RNA → ATAC): b_ji_rev = - ( (τ_i - τ_j + Δτ_ATAC,i)² / (2σ²) )
            # Here query = RNA (τ_j), key = ATAC (τ_i), diff = τ_i - τ_j.
            # The required offset is τ_i - τ_j + Δτ = diff + Δτ.
            offset = diff + delta_tau
        else:
            # Standard bias (ATAC → RNA): b_ij = - ( (τ_j - τ_i - Δτ_RNA,j)² / (2σ²) )
            # diff = τ_j - τ_i, offset = diff - Δτ.
            offset = diff - delta_tau

        bias = - (offset ** 2) / (2 * sigma_sq + 1e-8)
        # Expand heads dimension (same bias for all heads)
        bias = bias.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_kv)
        return bias

    def _windowed_sparsity_mask(
        self,
        tau_q: torch.Tensor,
        tau_kv: torch.Tensor,
        delta_tau: torch.Tensor,
        sigma_sq: torch.Tensor,
        bucket_size: int,
    ) -> torch.Tensor:
        """Create a mask for windowed attention (Algorithm 4).

        Returns a boolean mask where True indicates tokens that are within the
        temporal window and should be attended to.

        For simplicity, we implement a CPU-based version. For large-scale,
        we would bucket by pseudotime.

        Parameters
        ----------
        tau_q, tau_kv : torch.Tensor
            Pseudotimes (batch, seq_len).
        delta_tau : torch.Tensor
            Predicted lag per query-key pair (broadcastable).
        sigma_sq : torch.Tensor
            Window width squared.

        Returns
        -------
        torch.BoolTensor, shape (batch, seq_len_q, seq_len_kv)
            Mask with True for valid pairs.
        """
        # Compute window half-width = delta_tau_max + 3*sigma
        sigma = torch.sqrt(sigma_sq + 1e-8)
        half_window = self.delta_tau_max + 3 * sigma   # (batch, seq_len_q, seq_len_kv)

        tau_q_exp = tau_q.unsqueeze(-1)
        tau_kv_exp = tau_kv.unsqueeze(1)
        diff = tau_kv_exp - tau_q_exp
        within = torch.abs(diff - delta_tau) <= half_window
        return within

    def forward(
        self,
        query: torch.Tensor,          # (batch, seq_len_q, d_model)
        key: torch.Tensor,            # (batch, seq_len_kv, d_model)
        value: torch.Tensor,          # (batch, seq_len_kv, d_model)
        tau_q: torch.Tensor,          # (batch, seq_len_q)
        tau_kv: torch.Tensor,         # (batch, seq_len_kv)
        delta_tau: Optional[torch.Tensor] = None,  # (batch, seq_len_q, seq_len_kv) or (batch, seq_len_q)
        sigma_sq: Optional[torch.Tensor] = None,   # (batch, seq_len_q, seq_len_kv) or (batch, seq_len_q)
        reverse: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query : torch.Tensor
            Queries (e.g., ATAC tokens).
        key, value : torch.Tensor
            Keys and values (e.g., RNA tokens).
        tau_q, tau_kv : torch.Tensor
            Pseudotimes for query and key/value positions.
        delta_tau : torch.Tensor, optional
            Regulatory lag. If None, uses zero lag.
        sigma_sq : torch.Tensor, optional
            Window width squared. If None, uses a constant default.
        reverse : bool, default=False
            If True, uses reverse bias formula.
        attn_mask : torch.Tensor, optional
            Additional attention mask (e.g., for padding).

        Returns
        -------
        torch.Tensor, shape (batch, seq_len_q, d_model)
            Cross-attention output.
        """
        batch, seq_len_q, _ = query.shape
        seq_len_kv = key.shape[1]

        # Linear projections
        q = self.q_proj(query).reshape(batch, seq_len_q, self.n_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch, seq_len_kv, self.n_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch, seq_len_kv, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (batch, heads, seq_len_q, seq_len_kv)

        # Add lag bias if provided
        if delta_tau is not None:
            if delta_tau.dim() == 2:
                # Assume (batch, seq_len_q) -> broadcast over seq_len_kv
                delta_tau = delta_tau.unsqueeze(-1)  # (batch, seq_len_q, 1)
            if sigma_sq is None:
                sigma_sq = torch.ones_like(delta_tau) * 0.01
            elif sigma_sq.dim() == 2:
                sigma_sq = sigma_sq.unsqueeze(-1)
            bias = self._compute_bias(tau_q, tau_kv, delta_tau, sigma_sq, reverse=reverse)
            # Add bias to each head (bias has shape (batch, 1, seq_len_q, seq_len_kv))
            attn_logits = attn_logits + bias

        # Apply windowed sparsity mask if requested
        if self.use_windowed_sparsity and delta_tau is not None:
            if sigma_sq is None:
                sigma_sq = torch.ones_like(delta_tau) * 0.01
            window_mask = self._windowed_sparsity_mask(tau_q, tau_kv, delta_tau, sigma_sq, self.bucket_size)
            # Convert to attention mask: -inf where False
            window_mask = window_mask.unsqueeze(1)  # (batch, 1, seq_len_q, seq_len_kv)
            attn_logits = attn_logits.masked_fill(~window_mask, float('-inf'))

        # Apply additional attention mask (e.g., padding)
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = (attn_weights @ v).transpose(1, 2).reshape(batch, seq_len_q, self.d_model)
        out = self.out_proj(out)
        return out


class CrossAttentionPair(nn.Module):
    """Bidirectional cross-attention between ATAC and RNA with asymmetry.

    This module combines ATAC->RNA and RNA->ATAC cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.atac_to_rna = AsymmetricCrossAttention(d_model, n_heads, dropout, **kwargs)
        self.rna_to_atac = AsymmetricCrossAttention(d_model, n_heads, dropout, **kwargs)

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
        """Bidirectional cross-attention.

        Returns
        -------
        atac_out : torch.Tensor
            ATAC tokens after attending to RNA (reverse direction).
        rna_out : torch.Tensor
            RNA tokens after attending to ATAC.
        """
        # ATAC -> RNA (primary direction)
        rna_out = self.atac_to_rna(
            query=atac_tokens,
            key=rna_tokens,
            value=rna_tokens,
            tau_q=tau_atac,
            tau_kv=tau_rna,
            delta_tau=delta_tau_atac_to_rna,
            sigma_sq=sigma_sq_atac_to_rna,
            reverse=False,
            attn_mask=attn_mask,
        )
        # RNA -> ATAC (reverse direction)
        atac_out = self.rna_to_atac(
            query=rna_tokens,
            key=atac_tokens,
            value=atac_tokens,
            tau_q=tau_rna,
            tau_kv=tau_atac,
            delta_tau=delta_tau_rna_to_atac,
            sigma_sq=sigma_sq_rna_to_atac,
            reverse=True,
            attn_mask=attn_mask,
        )
        return atac_out, rna_out