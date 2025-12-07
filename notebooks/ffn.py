import marimo

__generated_with = "0.18.3"
app = marimo.App()


@app.cell
def _():
    import torch
    from torch.utils.data import DataLoader

    from boolrepr.models import FeedForwardNetwork
    from boolrepr.data import BooleanFunctionDataset

    return BooleanFunctionDataset, DataLoader, FeedForwardNetwork, torch


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
):
    train_dataset = BooleanFunctionDataset(
        num_samples=100,
        seq_length=seq_length,
        input_dim=input_dim,
        function_class=function_class,
    )

    train_loader = DataLoader(
        train_dataset.data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=BooleanFunctionDataset.collate_fn_feed_forward,
    )

    batch = next(iter(train_loader))
    batch["x"].shape, batch["y"].shape
    return (batch,)


@app.cell
def _(FeedForwardNetwork, device, input_dim, seq_length):
    model = FeedForwardNetwork(
        input_size=input_dim * seq_length,
        hidden_size=256,
        out_size=seq_length,
    )

    model = model.to(device)
    model
    return (model,)


@app.cell
def _(batch, model, torch):
    x = batch["x"]
    y = batch["y"]

    y_hat = model(x)

    loss = torch.nn.functional.cross_entropy(y, y_hat)
    loss.backward()
    return


if __name__ == "__main__":
    app.run()
