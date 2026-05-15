import torch
from tmo.models.attention import AsymmetricCrossAttention

def test_asymmetric_attention():
    batch, seq_q, seq_kv, d = 2, 5, 7, 64
    heads = 4
    attn = AsymmetricCrossAttention(d, heads)
    q = torch.randn(batch, seq_q, d)
    k = torch.randn(batch, seq_kv, d)
    v = torch.randn(batch, seq_kv, d)
    tau_q = torch.rand(batch, seq_q)
    tau_kv = torch.rand(batch, seq_kv)
    delta = torch.rand(batch, seq_q)
    out = attn(q, k, v, tau_q, tau_kv, delta_tau=delta, reverse=False)
    assert out.shape == (batch, seq_q, d)