"""Pseudotime inference utilities.

Wrappers for DPT and Palantir to compute continuous pseudotime along
differentiation trajectories.
"""

import numpy as np
import scanpy as sc
import anndata as ad
from typing import Optional, Union


def compute_dpt_pseudotime(
    adata: ad.AnnData,
    root_cell: Optional[Union[str, int]] = None,
    n_comps: int = 15,
) -> ad.AnnData:
    """Compute DPT pseudotime using Scanpy.

    Parameters
    ----------
    adata : AnnData
        Should have PCA computed (use RNA or LSI).
    root_cell : str or int, optional
        Cell index or barcode to use as root. If None, automatically determined.
    n_comps : int, default=15
        Number of diffusion components.

    Returns
    -------
    AnnData
        With .obs['dpt_pseudotime'] added (normalised to [0,1]).
    """
    # Ensure neighbors are computed
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_pca')
    # Diffusion map
    sc.tl.diffmap(adata, n_comps=n_comps)
    # DPT
    sc.tl.dpt(adata, n_branchings=0, root_key=None)
    if root_cell is not None:
        # Not straightforward; DPT uses root from diffmap. We can set
        sc.tl.dpt(adata, root_key=None, root_cell=root_cell)
    # Normalize to [0,1]
    pt = adata.obs['dpt_pseudotime']
    if pt.min() == pt.max():
        pt = np.zeros_like(pt)
    else:
        pt = (pt - pt.min()) / (pt.max() - pt.min())
    adata.obs['pseudotime'] = pt
    return adata


def compute_palantir_pseudotime(
    adata: ad.AnnData,
    start_cell: str,
    knn: int = 30,
    n_components: int = 20,
) -> ad.AnnData:
    """Compute pseudotime using Palantir.

    Requires Palantir package installed (github.com/dpeerlab/Palantir).

    Parameters
    ----------
    adata : AnnData
        With PCA or LSI already computed.
    start_cell : str
        Cell barcode of starting cell (e.g., stem cell).
    knn : int, default=30
        Number of nearest neighbours for graph.
    n_components : int, default=20
        Number of DMAP components.

    Returns
    -------
    AnnData
        With .obs['palantir_pseudotime'] and .obs['pseudotime'].
    """
    try:
        import palantir
    except ImportError:
        raise ImportError("Palantir not installed. Run: pip install palantir")

    # Use RNA PCA as basis
    dm_res = palantir.utils.run_diffusion_maps(adata.obsm['X_pca'], knn=knn, n_components=n_components)
    adata.obsm['X_dmap'] = dm_res['EigenVectors']
    adata.uns['palantir_dmap'] = dm_res

    # Determine pseudotime
    pt = palantir.core.run_palantir(
        adata.obsm['X_dmap'],
        start_cell=start_cell,
        num_waypoints=min(500, adata.n_obs),
    )
    adata.obs['palantir_pseudotime'] = pt['pseudotime']
    # Normalize
    pt_vals = adata.obs['palantir_pseudotime'].values
    pt_vals = (pt_vals - pt_vals.min()) / (pt_vals.max() - pt_vals.min())
    adata.obs['pseudotime'] = pt_vals
    return adata