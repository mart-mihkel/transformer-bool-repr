import torch

from torch import nn, Tensor


class NeuralNetwork(nn.Module):
    def __init__(
        self,
        in_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ParallelNetworks(nn.Module):
    def __init__(
        self,
        num_models: int,
        in_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        assert num_models > 0, f"Invalid number of networks: {num_models}"

        super(ParallelNetworks, self).__init__()

        self.nets = nn.ModuleList(
            [
                NeuralNetwork(
                    in_size=in_size,
                    hidden_size=hidden_size,
                    out_size=out_size,
                )
                for _ in range(num_models)
            ]
        )

    def forward(self, xs: Tensor) -> Tensor:
        assert len(self.nets) == xs.shape[0], "Invalid input shape"

        out = [net(x) for net, x in zip(self.nets, xs)]

        return torch.stack(out)
