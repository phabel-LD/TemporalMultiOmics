import torch
from tmo.models.lagmlp import LagMLP, CombinedLagWidthPredictor

def test_lagmlp():
    d = 64
    lag_mlp = LagMLP(d)
    z = torch.randn(10, d)
    e = torch.randn(10, d)
    out = lag_mlp(z, e)
    assert out.shape == (10,)
    assert out.min() >= -0.5
    assert out.max() <= 0.5

def test_combined():
    combined = CombinedLagWidthPredictor(d)
    z = torch.randn(10, d)
    e = torch.randn(10, d)
    dt, ss = combined(z, e)
    assert dt.shape == (10,)
    assert ss.shape == (10,)
    assert (ss > 0).all()