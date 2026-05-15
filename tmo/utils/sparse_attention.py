"""Utility functions for windowed sparse attention (Algorithm 4)."""

import torch
import numpy as np
from typing import List, Tuple


def bucket_by_pseudotime(
    pseudotime: torch.Tensor,
    bucket_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Sort cells by pseudotime and assign bucket IDs.

    Parameters
    ----------
    pseudotime : torch.Tensor, shape (n_cells,)
        Pseudotime values.
    bucket_size : int
        Number of cells per bucket.

    Returns
    -------
    sorted_indices : torch.Tensor
        Indices that sort pseudotime.
    bucket_ids : torch.Tensor, shape (n_cells,)
        Bucket ID for each cell (after sorting).
    bucket_boundaries : list of int
        Start and end indices for each bucket.
    """
    sorted_indices = torch.argsort(pseudotime)
    n_cells = len(pseudotime)
    n_buckets = (n_cells + bucket_size - 1) // bucket_size
    bucket_ids = torch.zeros(n_cells, dtype=torch.long)
    bucket_boundaries = []
    for b in range(n_buckets):
        start = b * bucket_size
        end = min(start + bucket_size, n_cells)
        bucket_ids[start:end] = b
        bucket_boundaries.append((start, end))
    return sorted_indices, bucket_ids, bucket_boundaries


def compute_neighbor_buckets(
    n_buckets: int,
    delta_tau_max: float,
    sigma_max: float,
    bucket_pseudotime_centroids: torch.Tensor,
) -> List[List[int]]:
    """Pre‑compute neighbor buckets based on pseudotime centroids and window size.

    Parameters
    ----------
    n_buckets : int
        Number of buckets.
    delta_tau_max : float
        Maximum absolute lag.
    sigma_max : float
        Maximum window width.
    bucket_pseudotime_centroids : torch.Tensor, shape (n_buckets,)
        Average pseudotime per bucket.

    Returns
    -------
    neighbor_buckets : list of list of int
        For each bucket, list of neighbor bucket indices.
    """
    delta = delta_tau_max + 3 * sigma_max
    neighbors = []
    for i in range(n_buckets):
        neighbors_i = []
        for j in range(n_buckets):
            if abs(bucket_pseudotime_centroids[j] - bucket_pseudotime_centroids[i]) <= delta:
                neighbors_i.append(j)
        neighbors.append(neighbors_i)
    return neighbors