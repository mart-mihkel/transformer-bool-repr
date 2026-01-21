import logging
from typing import Annotated

import einops
import torch
from torch import Tensor
from torch.nn import (
    LayerNorm,
    Linear,
    Module,
    ModuleList,
    Sequential,
    Sigmoid,
    Tanh,
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
            Tanh(),
        )
        self.out = Sequential(
            Linear(hidden_size, out_size),
            # Tanh()
            Sigmoid(),
        )

    def forward(
        self,
        x: Annotated[Tensor, "batch input"],
        return_layer: bool = False,
    ) -> tuple[Annotated[Tensor, "batch out"], Tensor | None]:
        hidden_layer = self.net(x)
        out = self.out(hidden_layer)
        if return_layer:
            return out, hidden_layer

        return out, None


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
        mask: Annotated[Tensor, "sequence embed"] | None = None,
    ) -> Annotated[Tensor, "batch sequence embed"]:
        q, k, v = self.proj_qkv(x).chunk(3, dim=-1)
        # q, k, v = x, x, x
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
        # self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn1 = Sequential(
            Linear(embed_dim, hidden_dim),
            Tanh(),
        )
        self.ffn2 = Linear(hidden_dim, embed_dim)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(
        self,
        x: Annotated[Tensor, "batch sequence embed"],
        mask: Annotated[Tensor, "sequence embed"] | None = None,
        return_layer: bool = False,
    ) -> tuple[Annotated[Tensor, "batch class"], Tensor | None]:
        res = x
        out = self.attn(x, mask)
        # out, _ = self.attn(x)
        out = self.norm1(out + res)

        res = out
        hidden_layer = self.ffn1(out)
        out = self.ffn2(hidden_layer)
        out = self.norm2(out + res)

        if return_layer:
            return out, hidden_layer

        return out, None


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

        self.fc = Linear(embed_dim, num_classes)
        self.sigmoid = Sigmoid()

    def forward(
        self,
        input_embeds: Annotated[Tensor, "batch sequence embed"],
        return_layer: bool = False,
    ) -> tuple[Annotated[Tensor, "batch class"], Tensor | None]:
        out = input_embeds
        for block in self.blocks:
            out, hidden_layer = block(out, return_layer=return_layer)

        if out.shape[1] == 1:
            if hidden_layer is not None:
                hidden_layer = hidden_layer[:, 0, :]

            out = out[:, 0, :]  # If sequence length = 1

        out2 = self.fc(out)
        out2 = self.sigmoid(out2)
        if return_layer:
            return out2, hidden_layer

        return out2, None
