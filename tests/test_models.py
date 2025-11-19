import torch

from boolrepr.models import ParallelNetworks


def test_parallel_networks():
    num_models = 5
    batch_size = 32
    in_size = 64
    out_size = 1

    model = ParallelNetworks(
        num_models=num_models,
        in_size=in_size,
        hidden_size=256,
        out_size=out_size,
    )

    x = torch.randn(num_models, batch_size, in_size)
    y = model.forward(x)

    assert y.shape == (num_models, batch_size, out_size)
