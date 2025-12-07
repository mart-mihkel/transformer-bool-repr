import logging
from typing import Annotated

import einops
import torch
from torch import Tensor
from torch.nn import (
    GELU,
    Embedding,
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    ReLU,
    Sequential,
)

logger = logging.getLogger("boolrepr")


class FeedForwardNetwork(Module):
    """
    FFN from [incontext bool](https://arxiv.org/abs/2310.03016)
    """

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
    """
    Parallel networks from [incontext bool](https://arxiv.org/abs/2310.03016)
    """

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


class MultiHeadAttention(Module):
    def __init__(self, d_embed: int, n_heads: int):
        super().__init__()
        assert d_embed % n_heads == 0, "Embedding shape not divisibe by number of heads"

        self.n_heads = n_heads
        self.scale = (d_embed // n_heads) ** -0.5

        self.proj_qkv = Linear(d_embed, d_embed * 3)
        self.proj_out = Linear(d_embed, d_embed)

    def forward(
        self,
        x: Annotated[Tensor, "batch sequence embed"],
    ) -> Annotated[Tensor, "batch sequence embed"]:
        q, k, v = self.proj_qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.head_partition, (q, k, v))

        attn_scores = q @ k.transpose(-1, -2) * self.scale
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


class TransformerEncoderBlock(Module):
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
        self, x: Annotated[Tensor, "batch sequence embed"]
    ) -> Annotated[Tensor, "batch sequence embed"]:
        res = x
        out = self.attn(x)
        out = self.norm1(out + res)

        res = out
        out = self.ffn(out)
        out = self.norm2(out + res)

        return out


class TransformerEncoder(Module):
    def __init__(
        self,
        voc_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_blocks: int,
        num_classes: int,
    ):
        super().__init__()

        self.embedding = Embedding(voc_size, embed_dim)
        self.blocks = ModuleList(
            [
                TransformerEncoderBlock(embed_dim, num_heads, hidden_dim)
                for _ in range(num_blocks)
            ]
        )

        self.classify = Linear(embed_dim, num_classes)

    def forward(
        self, x: Annotated[Tensor, "batch sequence"]
    ) -> Annotated[Tensor, "batch class"]:
        out = self.embedding(x)
        for block in self.blocks:
            out = block(out)

        out = out[:, 0, :]
        out = self.classify(out)
        out = torch.nn.functional.softmax(out, dim=-1)

        return out
