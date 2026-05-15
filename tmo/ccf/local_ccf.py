"""Local cross-correlation function (CCF) for estimating regulatory lags.

This module implements:
    - Sliding pseudotime window
    - Local CCF computation using interpolation onto a regular grid
      (non-uniform pseudotime handled via linear interpolation)
    - Block bootstrap (shift bootstrap) for significance testing
    - Optionally Gaussian process smoothing (via gp_smoothing.py)

The output is a signed lag surface Δτ_g(τ) of shape (n_genes, n_windows).
Positive lags → ATAC leads RNA (priming).
"""

import numpy as np
from scipy.signal import correlate
from scipy.interpolate import interp1d
from typing import Optional, Tuple
import warnings


def local_ccf_surface(
    atac_series: np.ndarray,
    rna_series: np.ndarray,
    pseudotime: np.ndarray,
    window_centers: np.ndarray,
    window_half_width: float = 0.1,
    max_lag: float = 0.3,
    n_lags: int = 61,
    bootstrap_samples: int = 0,
    block_length: int = 5,          # kept for interface compatibility, not used
    random_seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Estimate signed lag surface Δτ_g(τ) using local CCF.

    For each gene and each pseudotime window, the cross‑correlation is
    computed by first interpolating the ATAC and RNA series onto a dense
    regular pseudotime grid (200 points), then using `scipy.signal.correlate`.
    The peak lag is returned in pseudotime units.

    Parameters
    ----------
    atac_series : np.ndarray, shape (n_genes, n_cells)
        ATAC accessibility values (normalised) for each gene.
    rna_series : np.ndarray, shape (n_genes, n_cells)
        RNA expression values (normalised) for each gene.
    pseudotime : np.ndarray, shape (n_cells,)
        Pseudotime values for each cell, sorted in increasing order.
        Must be monotonic.
    window_centers : np.ndarray, shape (n_windows,)
        Pseudotime values at which to centre sliding windows.
    window_half_width : float, default=0.1
        Half-width of the sliding window in pseudotime units.
    max_lag : float, default=0.3
        Maximum absolute lag to consider (pseudotime units).
    n_lags : int, default=61
        Number of discrete lags to evaluate (used only in the returned
        `lags` grid; the actual peak search is over all computed lags).
    bootstrap_samples : int, default=0
        Number of bootstrap permutations for significance testing.
        If >0, p-values are computed via a shift bootstrap that
        preserves the temporal structure of the RNA series.
    block_length : int, default=5
        (Unused; kept for API compatibility.)
    random_seed : int, default=42
        Seed for the random number generator used in bootstrapping.

    Returns
    -------
    delta_tau : np.ndarray, shape (n_genes, n_windows)
        Estimated lag at each window center. Positive = ATAC leads RNA.
    p_values : np.ndarray, shape (n_genes, n_windows) or None
        P-value for each estimate (if bootstrap_samples > 0); otherwise None.
    lags : np.ndarray, shape (n_lags,)
        The lag values (in pseudotime units) that can be probed.
    """
    n_genes, n_cells = atac_series.shape
    if rna_series.shape != (n_genes, n_cells):
        raise ValueError("atac_series and rna_series must have same shape")
    if len(pseudotime) != n_cells:
        raise ValueError("pseudotime length must match number of cells")
    if not np.all(np.diff(pseudotime) >= 0):
        raise ValueError("pseudotime must be sorted in increasing order")

    # Precompute lag grid (for reference, not used in the peak search)
    lags = np.linspace(-max_lag, max_lag, n_lags)

    delta_tau = np.full((n_genes, len(window_centers)), np.nan)
    if bootstrap_samples > 0:
        p_values = np.full((n_genes, len(window_centers)), np.nan)
        rng = np.random.default_rng(random_seed)
    else:
        p_values = None

    for w_idx, tau0 in enumerate(window_centers):
        mask = (pseudotime >= tau0 - window_half_width) & (pseudotime <= tau0 + window_half_width)
        n_cells_window = np.sum(mask)
        if n_cells_window < 30:
            continue  # not enough cells, leave as NaN

        # Extract window data (already sorted because pseudotime is sorted)
        pt_win = pseudotime[mask]
        atac_win = atac_series[:, mask]
        rna_win = rna_series[:, mask]

        # Compute peak lag for each gene using the robust interpolation method
        for g in range(n_genes):
            try:
                best_lag, _, _ = ccf_single_gene(
                    atac_win[g], rna_win[g], pt_win,
                    max_lag=max_lag, n_lags=n_lags
                )
                delta_tau[g, w_idx] = best_lag
            except Exception:
                # In case of interpolation failure (e.g., constant values)
                delta_tau[g, w_idx] = np.nan

        # Bootstrap significance if requested
        if bootstrap_samples > 0:
            # Shift bootstrap: circularly shift the RNA series (in the regular
            # grid representation) and recompute peak lag. This preserves the
            # autocorrelation structure of RNA relative to ATAC.
            for g in range(n_genes):
                obs_lag = delta_tau[g, w_idx]
                if np.isnan(obs_lag):
                    p_values[g, w_idx] = np.nan
                    continue

                # Build the regular-grid representation that will be reused
                # We can do this once per gene per window.
                null_lags = []
                # Use a copy of the interpolation logic from ccf_single_gene
                # to avoid recomputing the grid each time.
                # We'll refactor by calling ccf_single_gene with a permuted RNA.
                # For permutation: create a new RNA array that is a circularly
                # shifted version of the original RNA sorted values.
                # Get original RNA sorted (already sorted by pseudotime)
                rna_sorted = rna_win[g]  # already in pseudotime order
                n_pts = len(pt_win)
                # Pre-interpolate to a regular grid? Actually we need to
                # recompute CCF for each shift. That is tolerable for small
                # bootstrap_samples and small windows.
                for b in range(bootstrap_samples):
                    # random shift between 1 and n_pts-1
                    shift = rng.integers(1, n_pts)
                    rna_shifted = np.roll(rna_sorted, shift)
                    # Compute CCF with shifted RNA
                    try:
                        null_lag, _, _ = ccf_single_gene(
                            atac_win[g], rna_shifted, pt_win,
                            max_lag=max_lag, n_lags=n_lags
                        )
                        null_lags.append(null_lag)
                    except Exception:
                        continue

                if len(null_lags) == 0:
                    p_values[g, w_idx] = np.nan
                else:
                    # Two-sided p-value: fraction of null lags with absolute
                    # value >= |obs_lag| (conservative for positive/negative)
                    null_abs = np.abs(null_lags)
                    p_values[g, w_idx] = np.mean(null_abs >= np.abs(obs_lag))

    return delta_tau, p_values, lags


def ccf_single_gene(
    atac: np.ndarray,
    rna: np.ndarray,
    pseudotime: np.ndarray,
    max_lag: float = 0.3,
    n_lags: int = 61,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute cross-correlation for a single gene using interpolation onto a regular pseudotime grid.

    Handles non-uniform pseudotime by linear interpolation to a dense uniform grid,
    then uses `scipy.signal.correlate` for accurate lag estimation.

    Parameters
    ----------
    atac : np.ndarray, shape (n_cells,)
        ATAC accessibility values (normalised).
    rna : np.ndarray, shape (n_cells,)
        RNA expression values (normalised).
    pseudotime : np.ndarray, shape (n_cells,)
        Pseudotime values, sorted.
    max_lag : float, default=0.3
        Maximum absolute lag to consider (pseudotime units).
    n_lags : int, default=61
        (Unused; kept for API compatibility.)

    Returns
    -------
    best_lag : float
        Lag at which cross-correlation is maximal (positive = ATAC leads RNA).
    lags_time : np.ndarray
        The pseudo‑time lags evaluated (within ±max_lag).
    correlations : np.ndarray
        Cross-correlation values at each lag.
    """
    # Ensure sorted
    sort_idx = np.argsort(pseudotime)
    pt = pseudotime[sort_idx]
    atac_sorted = atac[sort_idx]
    rna_sorted = rna[sort_idx]

    # Regular time grid with sufficient resolution
    n_regular = max(200, len(pt) * 2)  # at least 200 points
    pt_reg = np.linspace(pt.min(), pt.max(), n_regular)

    # Interpolate
    f_atac = interp1d(pt, atac_sorted, kind='linear', fill_value='extrapolate')
    f_rna = interp1d(pt, rna_sorted, kind='linear', fill_value='extrapolate')
    atac_reg = f_atac(pt_reg)
    rna_reg = f_rna(pt_reg)

    # Normalise (avoid division by zero)
    atac_std = np.std(atac_reg)
    rna_std = np.std(rna_reg)
    if atac_std == 0 or rna_std == 0:
        # Constant signal → no cross-correlation peak
        return np.nan, np.array([0.0]), np.array([0.0])
    atac_norm = (atac_reg - np.mean(atac_reg)) / atac_std
    rna_norm = (rna_reg - np.mean(rna_reg)) / rna_std

    # Compute full correlation
    corr = correlate(rna_norm, atac_norm, mode='full')
    lags_samples = np.arange(-len(atac_reg) + 1, len(atac_reg))
    dt_reg = pt_reg[1] - pt_reg[0]
    lags_time = lags_samples * dt_reg

    # Restrict to max_lag
    valid = np.abs(lags_time) <= max_lag
    if not np.any(valid):
        return np.nan, lags_time[valid], corr[valid]
    lags_time = lags_time[valid]
    corr = corr[valid]

    best_idx = np.argmax(corr)
    best_lag = lags_time[best_idx]

    return best_lag, lags_time, corr