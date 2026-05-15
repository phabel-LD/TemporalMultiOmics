import numpy as np
from tmo.ccf.local_ccf import ccf_single_gene, local_ccf_surface

def test_ccf_single_gene():
    t = np.linspace(0, 1, 50)
    atac = np.sin(2*np.pi*t) + 0.1*np.random.randn(50)
    rna = np.sin(2*np.pi*(t - 0.1)) + 0.1*np.random.randn(50)
    lag, lags, corr = ccf_single_gene(atac, rna, t, max_lag=0.3, n_lags=31)
    assert -0.3 <= lag <= 0.3
    assert len(lags) == 31

def test_local_ccf_surface():
    atac = np.random.randn(5, 200)
    rna = np.random.randn(5, 200)
    pt = np.sort(np.random.rand(200))
    centers = np.linspace(0.2, 0.8, 5)
    dt, pvals, lags = local_ccf_surface(atac, rna, pt, centers, window_half_width=0.1, max_lag=0.3, n_lags=21, bootstrap_samples=0)
    assert dt.shape == (5, 5)