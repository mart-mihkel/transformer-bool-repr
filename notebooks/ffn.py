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
    input_dim = 10
    batch_size = 64
    function_class = "conjunction"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return batch_size, device, function_class, input_dim


@app.cell
def _(BooleanFunctionDataset, function_class, input_dim):
    train_dataset = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
    )
    return (train_dataset,)


@app.cell
def _(DataLoader, batch_size, train_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    batch = next(iter(train_loader))
    batch
    return (batch,)


@app.cell
def _(FeedForwardNetwork, device, input_dim):
    model = FeedForwardNetwork(
        input_size=input_dim,
        hidden_size=256,
        out_size=1,
    )

    model = model.to(device)
    model
    return (model,)


@app.cell
def _(batch, model, torch):
    x = batch["x"]
    y = batch["y"]

    y_hat = model(x).flatten()
    y_hat = torch.nn.functional.sigmoid(y_hat)

    loss = torch.nn.functional.cross_entropy(y, y_hat)
    loss.backward()
    return


if __name__ == "__main__":
    app.run()
