import torch

from boolrepr.models import (
    ParallelFeedForwardNetworks,
    MultiHeadAttention,
    TransformerEncoder,
    TransformerEncoderBlock,
)


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


def test_multi_head_attention():
    batch, emb, seq, heads = 4, 8, 6, 4

    attn = MultiHeadAttention(d_embed=emb, n_heads=heads)
    x = torch.randn(batch, seq, emb)
    y = attn.forward(x)

    assert y.shape == (batch, seq, emb)


def test_transformer_encoder_block():
    batch, emb, seq, heads = 4, 8, 6, 4

    enc_block = TransformerEncoderBlock(
        embed_dim=emb,
        num_heads=heads,
        hidden_dim=seq,
    )

    x = torch.randn(batch, seq, emb)
    y = enc_block.forward(x)

    assert y.shape == (batch, seq, emb)


def test_transformer_encoder():
    voc, batch, emb, seq, heads, blocks, classes = 10, 4, 8, 6, 4, 12, 14

    enc = TransformerEncoder(
        voc_size=voc,
        embed_dim=emb,
        num_heads=heads,
        hidden_dim=seq,
        num_blocks=blocks,
        num_classes=classes,
    )

    x = torch.randint(low=1, high=voc, size=(batch, seq))
    y = enc.forward(x)

    assert y.shape == (batch, classes)
