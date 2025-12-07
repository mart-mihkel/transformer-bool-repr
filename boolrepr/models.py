"""
Model architectures from incontext bool paper: https://arxiv.org/abs/2310.03016
"""

import logging
from typing import Annotated

import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, ReLU, Sequential

logger = logging.getLogger("boolrepr")


class FeedForwardNetwork(Module):
    def __init__(
        self,
        in_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        super(FeedForwardNetwork, self).__init__()

        self.net = Sequential(
            Linear(in_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, out_size),
        )

    def forward(
        self, x: Annotated[Tensor, "batch input"]
    ) -> Annotated[Tensor, "batch out"]:
        return self.net(x)


class ParallelFeedForwardNetworks(Module):
    def __init__(
        self,
        num_models: int,
        in_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        assert num_models > 0, f"Invalid number of networks: {num_models}"

        super(ParallelFeedForwardNetworks, self).__init__()

        self.nets = ModuleList(
            [
                FeedForwardNetwork(
                    in_size=in_size,
                    hidden_size=hidden_size,
                    out_size=out_size,
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self, xs: Annotated[Tensor, "batch networks hidden"]
    ) -> Annotated[Tensor, "batch networks out"]:
        assert len(self.nets) == xs.shape[0], "Invalid input shape"

        out = [net(x) for net, x in zip(self.nets, xs)]

        return torch.stack(out)
