"""Gaussian process smoothing for the regulatory lag (Δτ) surface.

Provides functions to interpolate and smooth the signed lag surface Δτ_g(τ)
obtained from local CCF estimation. Uses Gaussian Process regression with
Matern-3/2 kernel to handle sparse windows and produce uncertainty estimates
(posterior variance) which are used downstream for inverse-variance weighting.

The raw Δτ surface obtained from local cross‑correlation can be noisy or
sparse (some windows may contain too few cells to estimate a reliable lag).
This module uses Gaussian Process (GP) regression with a Matern‑3/2 kernel
to:

  - interpolate missing values across pseudotime,
  - produce a smooth, continuous estimate of Δτ(τ) for each gene,
  - provide posterior uncertainty estimates (standard deviation) that can
    be used as inverse‑variance weights in downstream training.

A minimum of three observed points is required to fit a GP; genes with
fewer valid windows are left as NaN.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import Optional, Tuple


def smooth_delta_tau_surface(
    pseudotime_grid: np.ndarray,
    delta_tau_matrix: np.ndarray,
    gene_names: Optional[list] = None,
    length_scale_bounds: Tuple[float, float] = (1e-2, 0.5),
    noise_level_bounds: Tuple[float, float] = (1e-4, 1e0),
    return_uncertainty: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Smooth the Δτ surface using Gaussian process regression.

    For each gene, fit a GP to the observed Δτ values at the pseudotime
    window centers (where data exists), then predict at all pseudotime
    points of interest (typically a dense grid). Optionally returns
    posterior variance (uncertainty) for each prediction.

    Parameters
    ----------
    pseudotime_grid : np.ndarray, shape (n_windows,)
        Pseudotime coordinates of the window centers (from local CCF).
    delta_tau_matrix : np.ndarray, shape (n_genes, n_windows)
        Observed Δτ values (may contain NaNs for sparse windows).
    gene_names : list of str, optional
        Names of genes, used for debugging.
    length_scale_bounds : tuple, default=(1e-2, 1e2)
        Bounds for the length scale parameter of the Matern kernel.
    noise_level_bounds : tuple, default=(1e-4, 1e0)
        Bounds for the white noise kernel (observation noise).
    return_uncertainty : bool, default=True
        If True, return posterior standard deviation for each prediction.

    Returns
    -------
    smoothed_delta_tau : np.ndarray, shape (n_genes, n_predict)
        Smoothed Δτ values evaluated at the pseudotime grid points.
        The pseudotime grid is the same as input `pseudotime_grid` (non-NaN positions).
    uncertainty : np.ndarray, shape (n_genes, n_predict) or None
        Posterior standard deviation (if return_uncertainty=True), else None.
    """
    n_genes, n_win = delta_tau_matrix.shape

    # Identify windows with valid observations (non-NaN)
    valid_mask = ~np.isnan(delta_tau_matrix)

    # For each gene, extract valid points
    smoothed = np.full_like(delta_tau_matrix, np.nan)
    if return_uncertainty:
        unc = np.full_like(delta_tau_matrix, np.nan)
    else:
        unc = None

    # Fit a GP independently for each gene.
    for g in range(n_genes):
        valid_idx = valid_mask[g, :]
        # A meaningful GP requires at least three points.
        if np.sum(valid_idx) < 3:
            # Not enough points: keep NaN
            continue

        X_train = pseudotime_grid[valid_idx].reshape(-1, 1)
        y_train = delta_tau_matrix[g, valid_idx]

        # Kernel: Matern‑3/2 (provides a smooth yet flexible fit) plus a
        # white‑noise component that absorbs observational scatter.
        kernel = Matern(length_scale=0.1,
                        nu=1.5,
                        length_scale_bounds=length_scale_bounds) \
                 + WhiteKernel(noise_level=0.01,
                               noise_level_bounds=noise_level_bounds)
        gp = GaussianProcessRegressor(kernel=kernel,
                                      alpha=0.0, # no additional data‑level noise beyond the kernel
                                      normalize_y=True, # centre the target values before fitting
                                      random_state=42)
        gp.fit(X_train, y_train)

        # Predict at all pseudotime grid points
        X_pred = pseudotime_grid.reshape(-1, 1)
        y_pred, y_std = gp.predict(X_pred, return_std=True)
        smoothed[g, :] = y_pred
        if return_uncertainty:
            unc[g, :] = y_std

    if return_uncertainty:
        return smoothed, unc
    else:
        return smoothed, None


def estimate_uncertainty_weights(uncertainty: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute inverse-variance weights from GP uncertainty (posterior standard deviations).

    Used for weighting the LagMLP loss: w_{g,c} = 1 / (var + eps).

    These weights can be used to down‑weight contributions from genes or
    pseudotime windows where the GP fit is highly uncertain.  The formula
    is `w = 1 / (var + eps)`, where var is the squared standard deviation (sd^2)
    (the posterior variance).

    Parameters
    ----------
    uncertainty : np.ndarray, shape (n_genes, n_cells)
        Posterior standard deviation from GP (or any error estimate).
    eps : float, default=1e-6
        Small constant to avoid division by zero (when sd = 0).

    Returns
    -------
    weights : np.ndarray, shape (n_genes, n_cells)
        Inverse-variance weights (larger where uncertainty is small).
        Larger values correspond to lower uncertainty.
    """
    variance = uncertainty ** 2
    weights = 1.0 / (variance + eps)
    return weights