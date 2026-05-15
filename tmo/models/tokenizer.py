"""Tokenizer for converting single-cell multi-omics data into transformer token embeddings.

This module implements:
    - GeneVocab: mapping from gene/peak names to integer IDs.
    - ExpressionBinner: discretizes continuous expression/accessibility values into bins.
    - PseudotimeEmbedding: MLP that maps pseudotime scalar to a vector embedding.
    - Tokenizer: main class that produces input token embeddings for TMO from raw genes/peaks.
    - LatentTokenizer: tokenizer for pre‑computed latent components (PCA, LSI).

Token embeddings (LatentTokenizer) are:
    x_i = component_embedding(comp_idx) + modality_embedding(mod) + pseudotime_embedding(τ)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple


class GeneVocab:
    """Vocabulary for genes and ATAC peaks."""

    def __init__(self):
        self.gene_to_idx: Dict[str, int] = {}
        self.idx_to_gene: Dict[int, str] = {}
        self.next_idx: int = 0

    def add(self, name: str) -> int:
        if name not in self.gene_to_idx:
            idx = self.next_idx
            self.gene_to_idx[name] = idx
            self.idx_to_gene[idx] = name
            self.next_idx += 1
        return self.gene_to_idx[name]

    def __len__(self) -> int:
        return self.next_idx

    def get_idx(self, name: str) -> int:
        return self.gene_to_idx[name]

    def get_name(self, idx: int) -> str:
        return self.idx_to_gene[idx]


class ExpressionBinner(nn.Module):
    """Discretizes continuous expression/accessibility values into bins."""

    def __init__(self, n_bins: int = 10, method: str = 'quantile'):
        super().__init__()
        self.n_bins = n_bins
        self.method = method
        self.register_buffer('bin_edges', torch.zeros(n_bins + 1))
        self.fitted = False

    def fit(self, values: torch.Tensor):
        values_np = values.cpu().numpy().flatten()
        if self.method == 'quantile':
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            bin_edges = np.quantile(values_np, quantiles)
        elif self.method == 'width':
            log_vals = np.log10(values_np + 1e-6)
            min_val, max_val = log_vals.min(), log_vals.max()
            bin_edges = np.linspace(min_val, max_val, self.n_bins + 1)
            bin_edges = 10.0 ** bin_edges
        else:
            raise ValueError(f"Unknown binning method: {self.method}")
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.bin_edges = torch.tensor(bin_edges, dtype=torch.float32)
        self.fitted = True

    def forward(self, values: torch.Tensor) -> torch.LongTensor:
        assert self.fitted, "ExpressionBinner must be fit before use"
        bins = torch.bucketize(values, self.bin_edges, right=False) - 1
        bins = torch.clamp(bins, 0, self.n_bins - 1)
        return bins


class PseudotimeEmbedding(nn.Module):
    """Continuous pseudotime embedding using a small MLP."""

    def __init__(self, d_model: int, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        layers = []
        prev_dim = 1
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, d_model))
        self.mlp = nn.Sequential(*layers)

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        tau = tau.unsqueeze(-1)
        return self.mlp(tau)


class Tokenizer(nn.Module):
    """Tokenizer for raw gene/peak data (unused in latent pipeline)."""

    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 10,
        d_model: int = 512,
        binning_method: str = 'quantile',
        pseudotime_hidden_dims: List[int] = [128, 64],
    ):
        super().__init__()
        self.d_model = d_model
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.expr_embedding = nn.Embedding(n_bins, d_model)
        self.modality_embedding = nn.Embedding(2, d_model)
        self.pseudotime_embedding = PseudotimeEmbedding(d_model, pseudotime_hidden_dims)
        self.expr_binner = ExpressionBinner(n_bins, method=binning_method)
        self.binner_fitted = False

    def fit_binner(self, rna_values: torch.Tensor, atac_values: torch.Tensor):
        combined = torch.cat([rna_values.flatten(), atac_values.flatten()])
        self.expr_binner.fit(combined)
        self.binner_fitted = True

    def forward(
        self,
        gene_ids: torch.LongTensor,
        expression_values: torch.Tensor,
        pseudotime: torch.Tensor,
        modality: torch.LongTensor,
    ) -> torch.Tensor:
        batch, seq_len = gene_ids.shape
        gene_emb = self.gene_embedding(gene_ids)
        if not self.binner_fitted:
            raise RuntimeError("Binner not fitted.")
        bins = self.expr_binner(expression_values)
        expr_emb = self.expr_embedding(bins)
        mod_emb = self.modality_embedding(modality)
        tau_emb = self.pseudotime_embedding(pseudotime).unsqueeze(1).expand(-1, seq_len, -1)
        token_emb = gene_emb + expr_emb + mod_emb + tau_emb
        return token_emb


class LatentTokenizer(nn.Module):
    """Tokenizes cells using pre‑computed latent representations (e.g., PCA, LSI).

    Token embeddings = proj(latent_value) + modality_emb + pseudotime_emb
    """
    def __init__(self, n_rna_components, n_atac_components, d_model,
                 pseudotime_hidden_dims=[128, 64]):
        super().__init__()
        self.n_rna = n_rna_components
        self.n_atac = n_atac_components
        self.d_model = d_model

        # Project scalar latent value to d_model
        self.rna_proj = nn.Linear(1, d_model)
        self.atac_proj = nn.Linear(1, d_model)

        # Modality embeddings: 0 = RNA, 1 = ATAC
        self.modality_embedding = nn.Embedding(2, d_model)

        self.pseudotime_embedding = PseudotimeEmbedding(d_model, pseudotime_hidden_dims)

    def forward(self, rna_latent, atac_latent, pseudotime):
        batch = rna_latent.shape[0]
        device = rna_latent.device

        # RNA tokens
        rna_val = rna_latent.unsqueeze(-1)                # (batch, n_rna, 1)
        rna_emb = self.rna_proj(rna_val)                  # (batch, n_rna, d)
        rna_mod = self.modality_embedding(
            torch.zeros(batch, self.n_rna, dtype=torch.long, device=device))
        tau_emb = self.pseudotime_embedding(pseudotime).unsqueeze(1)  # (batch,1,d)
        rna_emb = rna_emb + rna_mod + tau_emb

        # ATAC tokens
        atac_val = atac_latent.unsqueeze(-1)
        atac_emb = self.atac_proj(atac_val)
        atac_mod = self.modality_embedding(
            torch.ones(batch, self.n_atac, dtype=torch.long, device=device))
        atac_emb = atac_emb + atac_mod + tau_emb

        tokens = torch.cat([rna_emb, atac_emb], dim=1)   # (batch, n_rna+n_atac, d)
        return tokens