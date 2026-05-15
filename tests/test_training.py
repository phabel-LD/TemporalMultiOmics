import torch
from tmo.training.losses import infonce_loss, lag_consistency_loss

def test_infonce():
    z1 = torch.randn(16, 128)
    z2 = torch.randn(16, 128)
    loss = infonce_loss(z1, z2)
    assert loss > 0

def test_lag_consistency():
    learned = torch.randn(100)
    prior = torch.randn(100)
    loss = lag_consistency_loss(learned, prior, prior_variance=0.01)
    assert loss > 0