"""Cell-state-conditional lag and width predictors (LagMLP and WidthMLP).

These modules map a concatenation of cell embedding (from ATAC) and gene/component
embedding to:
    - Δτ̂ : signed regulatory lag, bounded by [-Δτ_max, +Δτ_max]
    - σ̂² : positive window width, with a minimum value σ²_min

Both quantities are predicted from the concatenation of a cell embedding
(derived from the ATAC tokens in the first, unbiased forward pass) and a
component embedding that identifies which latent component the prediction
is made for.  The cell embedding provides cell‑state‑specific information,
while the component embedding allows the model to learn component‑specific
temporal offsets.

The architecture follows the manuscript exactly:
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
    
    The use of LayerNorm at the very beginning stabilises training and
    ensures that the two input embeddings (which may have different
    statistics) are brought to a common scale before the first linear
    transformation.

    The final tanh layer constrains the raw output to (-1, 1), which is
    then multiplied by Δτ_max (0.5) so that the predicted lag always lies
    in the biologically plausible range [-0.5, 0.5] pseudotime units.
    """

    def __init__(
        self,
        d_embed: int,
        hidden_dims: Optional[list] = None,
        delta_tau_max: float = 0.5,
        dropout: float = 0.1,
    ):
        """Initialise the LagMLP.

        Parameters
        ----------
        d_embed : int
            Dimensionality of the cell embedding and the component embedding
            (both are assumed to have the same dimension).
        hidden_dims : list of int, optional
            Sizes of the hidden layers.  Defaults to [d_embed, d_embed // 2].
        delta_tau_max : float, default=0.5
            Maximum absolute lag in pseudotime units.  The network's raw
            output is scaled by this value.
        dropout : float, default=0.1
            Dropout probability applied after each GELU activation.
        """

        super().__init__()
        if hidden_dims is None:
            hidden_dims = [d_embed, d_embed // 2]

        input_dim = 2 * d_embed # concatenated cell and component embedding
        self.norm = nn.LayerNorm(input_dim)

        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Tanh()) # output in [-1, 1]
        self.net = nn.Sequential(*layers)
        self.delta_tau_max = delta_tau_max

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        """Predict the signed regulatory lag for a given cell and component.

        Parameters
        ----------
        z_c : torch.Tensor, shape (batch, d_embed)
            Cell embedding obtained by mean‑pooling the ATAC token outputs
            after the first (unbiased) forward pass.
        e_g : torch.Tensor, shape (batch, d_embed)
            Learnable embedding of the component (ATAC or RNA) for which
            the lag is being predicted.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Predicted lag Δτ̂, clipped to [-Δτ_max, Δτ_max].
        """
        
        x = torch.cat([z_c, e_g], dim=-1) # (batch, 2*d_embed)
        x = self.norm(x)
        out = self.net(x).squeeze(-1) * self.delta_tau_max # (batch,)
        return out


class WidthMLP(nn.Module):
    """Multi‑layer perceptron that predicts a positive window width.

    The architecture is identical to LagMLP except for the final activation:
    instead of tanh, a Softplus function is used to ensure the output is
    strictly positive.  A minimum width σ²_min is added to avoid numerical
    issues when the width is used in the denominator of the attention bias.
    """

    def __init__(
        self,
        d_embed: int,
        hidden_dims: Optional[list] = None,
        sigma_sq_min: float = 0.001,
        dropout: float = 0.1,
    ):
        """Initialise the WidthMLP.

        Parameters
        ----------
        d_embed : int
            Dimensionality of the cell embedding and the component embedding.
        hidden_dims : list of int, optional
            Sizes of the hidden layers.  Defaults to [d_embed, d_embed // 2].
        sigma_sq_min : float, default=0.001
            Minimum value for the predicted squared width.  Ensures numerical
            stability when the width is used in the Gaussian attention bias.
        dropout : float, default=0.1
            Dropout probability.
        """

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

        # No activation on the final linear output; Softplus is applied in forward.
        self.net = nn.Sequential(*layers)
        self.sigma_sq_min = sigma_sq_min

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor) -> torch.Tensor:
        """Predict the window width squared for a given cell and component.

        Parameters
        ----------
        z_c : torch.Tensor, shape (batch, d_embed)
            Cell embedding.
        e_g : torch.Tensor, shape (batch, d_embed)
            Component embedding.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Predicted σ̂², guaranteed to be >= σ²_min.
        """

        x = torch.cat([z_c, e_g], dim=-1)
        x = self.norm(x)
        out = self.net(x).squeeze(-1) # raw output
        sigma_sq = self.sigma_sq_min + F.softplus(out)
        return sigma_sq


class CombinedLagWidthPredictor(nn.Module):
    """Convenience wrapper that bundles LagMLP and WidthMLP together.

    This module is used in the TMO model to produce both Δτ̂ and σ̂² for
    every component (ATAC and RNA) in every cell.  It optionally detaches
    the cell embedding before passing it to the two predictors, which can
    be useful during certain training phases (not used in the final pipeline
    where the cell embedding is never detached).
    """

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
        """Initialise the combined predictor.

        Parameters
        ----------
        d_embed : int
            Dimension of cell and component embeddings.
        lag_hidden_dims : list of int, optional
            Hidden layer sizes for the LagMLP.
        width_hidden_dims : list of int, optional
            Hidden layer sizes for the WidthMLP (defaults to same as LagMLP).
        delta_tau_max : float, default=0.5
            Maximum absolute lag.
        sigma_sq_min : float, default=0.001
            Minimum squared width.
        dropout : float, default=0.1
            Dropout probability for both MLPs.
        detach_cell_embedding : bool, default=False
            If True, the cell embedding is detached before being passed to
            the LagMLP and WidthMLP, blocking gradient flow into the encoder.
            This is used in the symmetric baseline or when the encoder is
            frozen; in the final asymmetric training it is set to False.
        """

        super().__init__()
        self.lag_mlp = LagMLP(d_embed, lag_hidden_dims, delta_tau_max, dropout)
        self.width_mlp = WidthMLP(d_embed, width_hidden_dims, sigma_sq_min, dropout)
        self.detach_cell_embedding = detach_cell_embedding

    def forward(self, z_c: torch.Tensor, e_g: torch.Tensor):
        """Predict lag and width for a batch of (cell, component) pairs.

        Parameters
        ----------
        z_c : torch.Tensor, shape (batch, d_embed)
            Cell embedding.
        e_g : torch.Tensor, shape (batch, d_embed)
            Component embedding.

        Returns
        -------
        delta_tau : torch.Tensor, shape (batch,)
            Predicted signed lag.
        sigma_sq : torch.Tensor, shape (batch,)
            Predicted positive window width squared.
        """
        
        if self.detach_cell_embedding:
            z_c = z_c.detach()
        delta_tau = self.lag_mlp(z_c, e_g)
        sigma_sq = self.width_mlp(z_c, e_g)
        return delta_tau, sigma_sq