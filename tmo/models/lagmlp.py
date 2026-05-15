"""Cell-state-conditional lag and width predictors (LagMLP and WidthMLP).

These modules map a concatenation of cell embedding (from ATAC) and gene/component
embedding to:
    - Δτ̂ : signed regulatory lag, bounded by [-Δτ_max, +Δτ_max]
    - σ̂² : positive window width, with a minimum value σ²_min

Architecture (matching manuscript):
    x = LayerNorm(concat(z_c, e_g))
    h1 = GELU(Linear(x -> d_embed))
    h2 = GELU(Linear(h1 -> d_embed // 2))
    output = appropriate activation (tanh for lag, Softplus for width)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LagMLP(nn.Module):
    """MLP that predicts signed regulatory lag from cell and gene embeddings.

    Architecture:
        - Concatenate cell embedding (z_c) and gene embedding (e_g)
        - LayerNorm (on the concatenated vector)
        - Linear -> GELU -> Dropout
        - Linear -> GELU -> Dropout
        - Linear -> Tanh (bounded output)
        - Scale by Δτ_max
    """

    def __init__(
        self,
        d_embed: int,
        hidden_dims: Optional[list] = None,
        delta_tau_max: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [d_embed, d_embed // 2]

        input_dim = 2 * d_embed
        self.norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.delta_tau_max = delta_tau_max

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_c, e_g], dim=-1)
        x = self.norm(x)
        out = self.net(x).squeeze(-1) * self.delta_tau_max
        return out


class WidthMLP(nn.Module):
    """MLP that predicts positive window width.

    Architecture matches LagMLP but with Softplus output to ensure positivity,
    plus a minimum width σ²_min.
    """

    def __init__(
        self,
        d_embed: int,
        hidden_dims: Optional[list] = None,
        sigma_sq_min: float = 0.001,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [d_embed, d_embed // 2]

        input_dim = 2 * d_embed
        self.norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)
        self.sigma_sq_min = sigma_sq_min

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_c, e_g], dim=-1)
        x = self.norm(x)
        out = self.net(x).squeeze(-1)
        sigma_sq = self.sigma_sq_min + F.softplus(out)
        return sigma_sq


class CombinedLagWidthPredictor(nn.Module):
    """Combines LagMLP and WidthMLP for convenience."""

    def __init__(
        self,
        d_embed: int,
        lag_hidden_dims: Optional[list] = None,
        width_hidden_dims: Optional[list] = None,
        delta_tau_max: float = 0.5,
        sigma_sq_min: float = 0.001,
        dropout: float = 0.1,
        detach_cell_embedding: bool = False,
    ):
        super().__init__()
        self.lag_mlp = LagMLP(d_embed, lag_hidden_dims, delta_tau_max, dropout)
        self.width_mlp = WidthMLP(d_embed, width_hidden_dims, sigma_sq_min, dropout)
        self.detach_cell_embedding = detach_cell_embedding

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor):
        if self.detach_cell_embedding:
            z_c = z_c.detach()
        delta_tau = self.lag_mlp(z_c, e_g)
        sigma_sq = self.width_mlp(z_c, e_g)
        return delta_tau, sigma_sq