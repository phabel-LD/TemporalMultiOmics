"""Utility functions for preprocessing."""

import scanpy as sc
import anndata as ad
import numpy as np
from scipy.sparse import issparse


def filter_genes_and_peaks(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    n_top_peaks: int = 20000,
    atac_key: str = 'ATAC',
) -> ad.AnnData:
    """
    Select highly variable genes and most frequent peaks.

    Parameters
    ----------
    adata : AnnData
        Must have RNA counts in .X and ATAC matrix in .obsm[atac_key].
    n_top_genes : int, default=2000
        Number of highly variable genes to keep.
    n_top_peaks : int, default=20000
        Number of most frequent peaks to keep.
    atac_key : str, default='ATAC'
        Key in .obsm where ATAC matrix is stored.

    Returns
    -------
    AnnData
        Same object, filtered in place.
    """
    # RNA: highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat')
    adata.var['highly_variable'] = adata.var.get('highly_variable', False)
    adata._inplace_subset_var(adata.var['highly_variable'])

    # ATAC: keep most frequent peaks
    X = adata.obsm[atac_key]
    if issparse(X):
        total = np.asarray(X.sum(axis=0)).ravel()
    else:
        total = X.sum(axis=0)
    top_idx = np.argsort(-total)[:n_top_peaks]
    adata.obsm[atac_key] = X[:, top_idx]
    # Also update stored peak names if present
    if 'peak_names' in adata.uns:
        adata.uns['peak_names'] = [adata.uns['peak_names'][i] for i in top_idx]
    return adata