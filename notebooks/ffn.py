import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import torch
    from torch.utils.data import DataLoader

    from boolrepr.models import FeedForwardNetwork
    from boolrepr.data import BooleanFunctionDataset

    return BooleanFunctionDataset, DataLoader, FeedForwardNetwork, mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bogus
    """)
    return


@app.cell
def _(torch):
    input_dim = 32
    seq_length = 64
    batch_size = 64
    function_class = "conjunction"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return batch_size, device, function_class, input_dim, seq_length


@app.cell
def _(
    BooleanFunctionDataset,
    DataLoader,
    batch_size,
    function_class,
    input_dim,
    seq_length,
    torch,
):
    train_dataset = BooleanFunctionDataset(
        num_samples=10_000,
        seq_length=seq_length,
        input_dim=input_dim,
        function_class=function_class,
    )

    train_loader = DataLoader(
        train_dataset.data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: torch.stack(x),
    )
    return (train_loader,)


@app.cell
def _(FeedForwardNetwork, device, input_dim, seq_length):
    model = FeedForwardNetwork(
        in_size=input_dim * seq_length * 2,
        hidden_size=256,
        out_size=1,
    )

    model = model.to(device)
    return


@app.cell
def _(train_loader):
    batch = next(iter(train_loader))
    batch.shape
    return


if __name__ == "__main__":
    app.run()
