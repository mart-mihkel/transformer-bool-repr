import torch

from boolrepr.models import (
    ParallelFeedForwardNetworks,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerBlock,
)


def test_parallel_networks():
    batch, n_nets, hidden, d_in, d_out = 4, 2, 8, 6, 1

    model = ParallelFeedForwardNetworks(
        num_models=n_nets,
        input_size=d_in,
        hidden_size=hidden,
        out_size=d_out,
    )

    x = torch.randn(batch, d_in)
    y = model.forward(x)

    assert y.shape == (batch, d_out)


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
    y = enc_block.forward(x)

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
    y = enc.forward(x)

    assert y.shape == (batch, classes)
