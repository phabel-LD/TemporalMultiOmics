import pytest
import numpy as np
import anndata as ad
from tmo.data.loader import load_mouse_brain_processed, filter_genes_and_peaks

def test_load_mouse_brain_processed(tmp_path):
    # Create dummy AnnData
    adata = ad.AnnData(np.random.rand(100, 200))
    adata.obs['pseudotime'] = np.linspace(0, 1, 100)
    adata.obsm['ATAC'] = np.random.rand(100, 500)
    path = tmp_path / "test.h5ad"
    adata.write(path)
    loaded = load_mouse_brain_processed(path)
    assert loaded.n_obs == 100
    assert 'pseudotime' in loaded.obs
    assert 'ATAC' in loaded.obsm