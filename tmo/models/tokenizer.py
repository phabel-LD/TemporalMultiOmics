"""Tokenizer for converting single-cell multi-omics data into transformer token embeddings.

This module provides the embedding layer that converts pre‑computed latent
representations (PCA for RNA, LSI for ATAC) into the token sequences consumed
by the TMO transformer.

The central class is ``LatentTokenizer``, which is used in the final pipeline.
It produces token embeddings of the form

    x = proj(latent_value) + modality_embedding(mod) + pseudotime_embedding(τ)

where ``proj`` is a learned linear projection of the scalar PCA or LSI value,
``modality_embedding`` distinguishes RNA (0) from ATAC (1), and
``pseudotime_embedding`` maps the cell's pseudotime coordinate to a vector.

The older ``Tokenizer`` class, which operates on raw gene/peak counts, is
retained for completeness but is not used in the publication.

Supporting utilities:
    - ``GeneVocab``: maps gene/peak names to integer indices.
    - ``ExpressionBinner``: discretises continuous values into bins for the
      raw‑data tokenizer.
    - ``PseudotimeEmbedding``: a small MLP that embeds the scalar pseudotime.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict, Tuple


class GeneVocab:
    """Vocabulary that assigns a unique integer index to every gene or peak name.

    This is a simple two‑way mapping used by the raw‑data ``Tokenizer``.
    It is not required by the latent pipeline.
    """

    def __init__(self):
        self.gene_to_idx: Dict[str, int] = {}
        self.idx_to_gene: Dict[int, str] = {}
        self.next_idx: int = 0

    def add(self, name: str) -> int:
        """Register a gene/peak name.

        Parameters
        ----------
        name : str
            The name to add.

        Returns
        -------
        int
            The index assigned to this name.
        """

        if name not in self.gene_to_idx:
            idx = self.next_idx
            self.gene_to_idx[name] = idx
            self.idx_to_gene[idx] = name
            self.next_idx += 1
        return self.gene_to_idx[name]

    def __len__(self) -> int:
        """Return the number of unique names in the vocabulary."""
        return self.next_idx

    def get_idx(self, name: str) -> int:
        """Look up the index of a name.

        Raises ``KeyError`` if the name is not present.
        """
        return self.gene_to_idx[name]

    def get_name(self, idx: int) -> str:
        """Return the name corresponding to a given index.

        Raises ``KeyError`` if the index is not present.
        """
        return self.idx_to_gene[idx]


class ExpressionBinner(nn.Module):
    """Discretises continuous expression or accessibility values into bins.

    For the raw‑data ``Tokenizer``, continuous values are first binned and
    then embedded.  This class learns the bin edges from data (using quantiles
    or equal‑width intervals) and, during forward, maps each value to a bin
    index.

    Not used in the latent pipeline.
    """

    def __init__(self, n_bins: int = 10, method: str = 'quantile'):
        """Initialise the binner.

        Parameters
        ----------
        n_bins : int, default=10
            Number of bins.
        method : str, default='quantile'
            Binning strategy.  ``'quantile'`` assigns bins so that each
            contains approximately the same number of data points;
            ``'width'`` uses equal intervals on a log10 scale.
        """
        
        super().__init__()
        self.n_bins = n_bins
        self.method = method
        self.register_buffer('bin_edges', torch.zeros(n_bins + 1))
        self.fitted = False

    def fit(self, values: torch.Tensor):
        """Estimate bin edges from data.

        Parameters
        ----------
        values : torch.Tensor, shape (n_cells, n_features)
            Expression or accessibility values to use for fitting.
        """

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
        # The first and last edges are set to ±∞ so that every value falls inside a bin.
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        self.bin_edges = torch.tensor(bin_edges, dtype=torch.float32)
        self.fitted = True

    def forward(self, values: torch.Tensor) -> torch.LongTensor:
        """Bin the input values.

        Parameters
        ----------
        values : torch.Tensor
            Values to bin.

        Returns
        -------
        torch.LongTensor
            Bin indices (0‑based) with the same shape as the input.
        """

        assert self.fitted, "ExpressionBinner must be fit before use"
        bins = torch.bucketize(values, self.bin_edges, right=False) - 1
        bins = torch.clamp(bins, 0, self.n_bins - 1)
        return bins


class PseudotimeEmbedding(nn.Module):
    """Continuous pseudotime embedding using a small MLP.

    Maps a scalar pseudotime value in [0, 1] to a ``d_model``‑dimensional
    vector that is added to every token embedding of a cell.
    """

    def __init__(self, d_model: int, hidden_dims: List[int] = [128, 64]):
        """Initialise the MLP.

        Parameters
        ----------
        d_model : int
            Output embedding dimension.
        hidden_dims : list of int, default=[128, 64]
            Sizes of the hidden layers.  The first hidden layer receives a
            1‑dimensional input; the last hidden layer must project to
            ``d_model``.
        """

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
        """Embed pseudotime values.

        Parameters
        ----------
        tau : torch.Tensor, shape (batch,)
            Pseudotime values in [0, 1].

        Returns
        -------
        torch.Tensor, shape (batch, d_model)
        """

        tau = tau.unsqueeze(-1) # (batch, 1)
        return self.mlp(tau)


class Tokenizer(nn.Module):
    """Tokenizer for raw gene/peak data.

    This class embeds each feature as the sum of a gene embedding, an
    expression‑bin embedding, a modality embedding, and a pseudotime
    embedding.  It requires a pre‑fitted ``ExpressionBinner``.

    **Not used in the latent pipeline.**  It is kept for historical
    completeness and may be useful for future extensions that operate
    directly on count matrices.
    """

    def __init__(
        self,
        vocab_size: int,
        n_bins: int = 10,
        d_model: int = 512,
        binning_method: str = 'quantile',
        pseudotime_hidden_dims: List[int] = [128, 64],
    ):
        """Initialise the raw‑data tokenizer.

        Parameters
        ----------
        vocab_size : int
            Total number of gene + peak tokens.
        n_bins : int, default=10
            Number of expression/accessibility bins.
        d_model : int, default=512
            Dimension of token embeddings.
        binning_method : str, default='quantile'
            Binning method for ``ExpressionBinner``.
        pseudotime_hidden_dims : list of int, default=[128, 64]
            Hidden dimensions for ``PseudotimeEmbedding``.
        """

        super().__init__()
        self.d_model = d_model
        self.gene_embedding = nn.Embedding(vocab_size, d_model)
        self.expr_embedding = nn.Embedding(n_bins, d_model)
        # Modality embeddings: 0 = RNA, 1 = ATAC
        self.modality_embedding = nn.Embedding(2, d_model)
        self.pseudotime_embedding = PseudotimeEmbedding(d_model, pseudotime_hidden_dims)

        self.expr_binner = ExpressionBinner(n_bins, method=binning_method)
        self.binner_fitted = False

    def fit_binner(self, rna_values: torch.Tensor, atac_values: torch.Tensor):
        """Fit the expression binner on RNA and ATAC values.

        Parameters
        ----------
        rna_values : torch.Tensor, shape (n_cells, n_genes)
        atac_values : torch.Tensor, shape (n_cells, n_peaks)
        """

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
        """Produce token embeddings from raw inputs.

        Parameters
        ----------
        gene_ids : torch.LongTensor, shape (batch, seq_len)
            Gene or peak IDs for each token.
        expression_values : torch.Tensor, shape (batch, seq_len)
            Expression or accessibility value for each token.
        pseudotime : torch.Tensor, shape (batch,)
            Pseudotime of each cell (same for all tokens of a cell).
        modality : torch.LongTensor, shape (batch, seq_len)
            Modality indicator: 0 = RNA, 1 = ATAC.

        Returns
        -------
        torch.Tensor, shape (batch, seq_len, d_model)
        """

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
    """Tokenizes cells using pre‑computed latent representations (PCA for RNA, LSI for ATAC).

    This is the tokenizer used in the final TMO pipeline.  For each cell it
    creates a fixed sequence of tokens: the first ``n_rna`` tokens correspond
    to the RNA latent components, the next ``n_atac`` tokens to the ATAC
    latent components.

    Token embedding formula
    -----------------------
    For a component with latent value ``v``, modality ``m`` (0 = RNA, 1 = ATAC),
    and pseudotime ``τ``, the embedding is

        x = proj(v) + modality_embedding(m) + pseudotime_embedding(τ)

    where ``proj`` is a learned linear layer that maps the scalar latent value
    to ``d_model`` dimensions.  This design gives every token cell‑state‑specific
    information, which is essential for the model to learn meaningful cell
    embeddings and regulatory lags.

    Parameters
    ----------
    n_rna_components : int
        Number of RNA latent components (e.g., 50).
    n_atac_components : int
        Number of ATAC latent components (e.g., 50).
    d_model : int
        Dimension of the output token embeddings.
    pseudotime_hidden_dims : list of int, default=[128, 64]
        Hidden layer sizes for the ``PseudotimeEmbedding`` MLP.
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
        """Create token embeddings for a batch of cells.

        Parameters
        ----------
        rna_latent : torch.Tensor, shape (batch, n_rna)
            RNA PCA components for each cell.
        atac_latent : torch.Tensor, shape (batch, n_atac)
            ATAC LSI components for each cell.
        pseudotime : torch.Tensor, shape (batch,)
            Pseudotime value for each cell, in [0, 1].

        Returns
        -------
        torch.Tensor, shape (batch, n_rna + n_atac, d_model)
            The token sequence, RNA tokens first, then ATAC tokens.
        """
        batch = rna_latent.shape[0]
        device = rna_latent.device

        # ---- RNA tokens ----
        rna_val = rna_latent.unsqueeze(-1)                # (batch, n_rna, 1)
        rna_emb = self.rna_proj(rna_val)                  # (batch, n_rna, d)
        rna_mod = self.modality_embedding(
            torch.zeros(batch, self.n_rna, dtype=torch.long, device=device)) # (batch, n_rna, d)
        tau_emb = self.pseudotime_embedding(pseudotime).unsqueeze(1)  # (batch, 1, d)
        rna_emb = rna_emb + rna_mod + tau_emb # (batch, n_rna, d)

        # ---- ATAC tokens ----
        atac_val = atac_latent.unsqueeze(-1)
        atac_emb = self.atac_proj(atac_val)
        atac_mod = self.modality_embedding(
            torch.ones(batch, self.n_atac, dtype=torch.long, device=device))
        atac_emb = atac_emb + atac_mod + tau_emb # (batch, n_atac, d)

        # ---- Concatenate: RNA first, then ATAC ----
        tokens = torch.cat([rna_emb, atac_emb], dim=1)   # (batch, n_rna + n_atac, d)
        return tokens