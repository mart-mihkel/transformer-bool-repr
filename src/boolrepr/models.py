import logging
from typing import Annotated

import einops
import torch
from torch import Tensor
from torch.nn import (
    GELU,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
    Sigmoid,
)

logger = logging.getLogger("boolrepr")


class FeedForwardNetwork(Module):
    """
    FFN from [incontext bool](https://arxiv.org/abs/2310.03016)
    """

    def __init__(
        self,
        input_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        super(FeedForwardNetwork, self).__init__()

        self.net = Sequential(
            Linear(input_size, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size),
            ReLU(),
            Linear(hidden_size, out_size),
            Sigmoid(),
        )

    def forward(
        self, x: Annotated[Tensor, "batch input"]
    ) -> Annotated[Tensor, "batch out"]:
        return self.net(x)


class ParallelFeedForwardNetworks(Module):
    """
    Parallel networks from [incontext bool](https://arxiv.org/abs/2310.03016)
    """

    def __init__(
        self,
        num_models: int,
        input_size: int = 64,
        hidden_size: int = 256,
        out_size: int = 1,
    ):
        assert num_models > 0, "Non-positive number of networks"

        super(ParallelFeedForwardNetworks, self).__init__()

        self.num_models = num_models
        self.nets = ModuleList(
            [
                FeedForwardNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    out_size=out_size,
                )
                for _ in range(num_models)
            ]
        )

    def forward(
        self, x: Annotated[Tensor, "batch input"]
    ) -> Annotated[Tensor, "batch out"]:
        b = x.shape[0]
        n = self.num_models

        assert b % n == 0, "Batch size not divisib by number of networks"

        chunks = x.view(n, b // n, -1)
        out = [net(c) for net, c in zip(self.nets, chunks)]
        return torch.cat(out, dim=0)


class MultiHeadAttention(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, (
            "Embedding shape not divisibe by number of heads"
        )

        self.n_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.proj_qkv = Linear(embed_dim, embed_dim * 3)
        self.proj_out = Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: Annotated[Tensor, "batch sequence embed"],
        mask: Annotated[Tensor, "sequence sequence"] | None = None,
    ) -> Annotated[Tensor, "batch sequence embed"]:
        q, k, v = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = q @ k.transpose(-1, -2) * self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)

        out = attn_weights @ v
        out = self.head_merge(out)
        out = self.proj_out(out)

        return out

    def head_partition(
        self, x: Annotated[Tensor, "batch sequence embed"]
    ) -> Annotated[Tensor, "batch head sequence head_embed"]:
        return einops.rearrange(x, "b s (h d) -> b h s d", h=self.n_heads)

    def head_merge(
        self, x: Annotated[Tensor, "batch head sequence head_embed"]
    ) -> Annotated[Tensor, "batch sequence embed"]:
        return einops.rearrange(x, "b h s d -> b s (h d)")


class TransformerBlock(Module):
    def __init__(self, embed_dim: int, num_heads: int, hidden_dim: int):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = Sequential(
            Linear(embed_dim, hidden_dim),
            GELU(),
            Linear(hidden_dim, embed_dim),
        )

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(
        self,
        x: Annotated[Tensor, "batch sequence embed"],
        mask: Annotated[Tensor, "sequence sequence"] | None = None,
    ) -> Annotated[Tensor, "batch sequence embed"]:
        res = x
        out = self.attn(x, mask)
        out = self.norm1(out + res)

        res = out
        out = self.ffn(out)
        out = self.norm2(out + res)

        return out


class TransformerEncoder(Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int,
        num_classes: int,
    ):
        super().__init__()

        self.blocks = ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        self.classify = Sequential(Linear(embed_dim, num_classes), Sigmoid())

    def forward(
        self,
        input_ids: Annotated[Tensor, "batch sequence"],
    ) -> Annotated[Tensor, "batch class"]:
        out = input_ids

        for block in self.blocks:
            out = block(out)

        out = out[:, 0, :]
        out = self.classify(out)

        return out
