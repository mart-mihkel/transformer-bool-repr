import torch

from boolrepr.models import (
    MultiHeadAttention,
    TransformerEncoder,
    TransformerBlock,
)


def test_multi_head_attention():
    batch, emb, seq, heads = 10, 8, 6, 4

    attn = MultiHeadAttention(embed_dim=emb, num_heads=heads)
    x = torch.randn(batch, seq, emb)
    y = attn.forward(x)

    assert y.shape == (batch, seq, emb)


def test_transformer_block():
    batch, emb, seq, heads = 10, 8, 6, 4

    enc_block = TransformerBlock(
        embed_dim=emb,
        num_heads=heads,
        hidden_dim=seq,
    )

    x = torch.randn(batch, seq, emb)
    y, _ = enc_block.forward(x)

    assert y.shape == (batch, seq, emb)


def test_transformer_encoder():
    batch, emb, seq, heads, blocks, classes = 2, 8, 6, 4, 12, 14

    enc = TransformerEncoder(
        embed_dim=emb,
        num_heads=heads,
        hidden_dim=seq,
        num_blocks=blocks,
        num_classes=classes,
    )

    x = torch.randint(low=0, high=1, size=(batch, 1, emb)).float()
    y, _ = enc.forward(x)

    assert y.shape == (batch, classes)
