import torch

from boolrepr.models import ParallelFeedForwardNetworks


def test_parallel_networks():
    num_models = 2
    batch_size = 4
    in_size = 6
    out_size = 1

    model = ParallelFeedForwardNetworks(
        num_models=num_models,
        in_size=in_size,
        hidden_size=2,
        out_size=out_size,
    )

    x = torch.randn(num_models, batch_size, in_size)
    y = model.forward(x)

    assert y.shape == (num_models, batch_size, out_size)


# def test_self_attention():
#     b, n, h, e = 32, 64, 12, 48
#     q, k, v = torch.randn(3, b, n, h, e // h)
#     attn = TransformerBlock.self_attention(q, k, v)
#     assert attn.shape == (b, n, h, e // h)
