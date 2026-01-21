import marimo

__generated_with = "0.19.4"
app = marimo.App()

with app.setup:
    from boolrepr.clustering import Clustering
    from boolrepr.scripts.train_ffn import main as train_ffn

    import matplotlib.pyplot as plt


@app.cell
def _():
    function_class = "conjunction"
    input_dim = 8
    epochs = 25
    batch_size = 128
    parity_relevant_vars = 2
    hidden_dim = 64
    train_data_proportion = 0.8
    out_dir = "out/ffn"
    random_seed = None
    return (
        batch_size,
        epochs,
        function_class,
        hidden_dim,
        input_dim,
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
    out_dir,
    parity_relevant_vars,
    random_seed,
    train_data_proportion,
):
    trainer, func, model = train_ffn(
        function_class=function_class,
        input_dim=input_dim,
        epochs=epochs,
        batch_size=batch_size,
        parity_relevant_vars=parity_relevant_vars,
        hidden_dim=hidden_dim,
        train_data_proportion=train_data_proportion,
        out_dir=out_dir,
        random_seed=random_seed,
    )
    return func, model, trainer


@app.cell
def _(epochs, func, model, out_dir, trainer):
    testing_epochs = list(range(1, epochs + 1, max(epochs // 50, 1)))

    cluster = Clustering(
        model,
        out_dir,
        testing_epochs,
        trainer.eval_loader,
        trainer.fourier_coefs,
        func.relevant_vars,
    )

    cluster.test_ood(model)
    # cluster.correlate()
    clusters_per_epoch = cluster.cluster_over_epochs()

    cluster.visualize(
        clusters_per_epoch,
        [
            item["eval_accuracy"]
            for item in trainer.telemetry
            if item["epoch"] in testing_epochs
        ],
        [
            item["train_accuracy"]
            for item in trainer.telemetry
            if item["epoch"] in testing_epochs
        ],
        "figure_FFN.pdf",
    )

    plt.show()
    return


if __name__ == "__main__":
    app.run()
