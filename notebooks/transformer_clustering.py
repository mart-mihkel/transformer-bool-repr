import marimo

__generated_with = "0.19.4"
app = marimo.App()

with app.setup:
    from boolrepr.clustering import Clustering
    from boolrepr.scripts.train_transformer import main as train_transformer

    import polars as pl
    import seaborn as sns
    import matplotlib.pyplot as plt


@app.cell
def _():
    function_class = "parity"
    input_dim = 10
    epochs = 200
    batch_size = 1024
    parity_relevant_vars = 3
    num_blocks = 3
    num_heads = 1
    hidden_dim = 128
    train_data_proportion = 0.8
    out_dir = "out/transformer"
    random_seed = 2
    return (
        batch_size,
        epochs,
        function_class,
        hidden_dim,
        input_dim,
        num_blocks,
        num_heads,
        out_dir,
        parity_relevant_vars,
        random_seed,
        train_data_proportion,
    )


@app.cell
def _(
    batch_size,
    epochs,
    function_class,
    hidden_dim,
    input_dim,
    num_blocks,
    num_heads,
    out_dir,
    parity_relevant_vars,
    random_seed,
    train_data_proportion,
):
    trainer, func, model = train_transformer(
        function_class=function_class,
        input_dim=input_dim,
        epochs=epochs,
        batch_size=batch_size,
        parity_relevant_vars=parity_relevant_vars,
        num_blocks=num_blocks,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        train_data_proportion=train_data_proportion,
        out_dir=out_dir,
        random_seed=random_seed,
    )
    return func, model, trainer


@app.cell
def _(epochs, func, model, out_dir, trainer):
    _testing_epochs = list(range(1, epochs + 1, max(1, epochs // 50)))

    _cluster = Clustering(
        model,
        out_dir,
        _testing_epochs,
        trainer.eval_loader,
        trainer.fourier_coefs,
        func.relevant_vars,
    )

    _cluster.test_ood(model)
    # _cluster.correlate()
    _clusters_per_epoch = _cluster.cluster_over_epochs()

    _eval_accs = [
        item["eval_accuracy"]
        for item in trainer.telemetry
        if item["epoch"] in _testing_epochs
    ]

    _train_accs = [
        item["train_accuracy"]
        for item in trainer.telemetry
        if item["epoch"] in _testing_epochs
    ]

    df = pl.DataFrame(
        data={
            "Epoch": _clusters_per_epoch.keys(),
            "Clusters": _clusters_per_epoch.values(),
            "Train": _train_accs,
            "Eval": _eval_accs,
        }
    )

    df
    return (df,)


@app.cell
def _(df):
    sns.set_style("white")
    sns.set_context("talk")
    palette = sns.color_palette("ch:s=.25,rot=-.25")
    palette2 = sns.color_palette("flare")
    fig, ax1 = plt.subplots(figsize=(8, 7))
    ax2 = ax1.twinx()

    sns.lineplot(
        data=df,
        x="Epoch",
        y="Clusters",
        ax=ax1,
        label="Clusters",
        color=palette2[0],
    )

    sns.lineplot(
        data=df,
        x="Epoch",
        y="Train",
        ax=ax2,
        label="Train accuracy",
        color=palette[1],
    )

    sns.lineplot(
        data=df,
        x="Epoch",
        y="Eval",
        ax=ax2,
        label="Eval accuracy",
        color=palette[2],
    )

    # sns.despine(offset=10, trim=True);
    ax2.tick_params(axis="y", right=False)
    ax1.tick_params(axis="y", left=False)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")

    ax1.set_ylabel("Clusters")
    ax1.set_yticks(range(0, df["Clusters"].max() + 1, 2))
    ax1.set_xlabel("Epoch")
    ax1.legend(loc="lower left")

    plt.savefig("transformer-clustering.pdf", transparent=True)
    plt.show()
    return


@app.cell
def _(model):
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    return


if __name__ == "__main__":
    app.run()
