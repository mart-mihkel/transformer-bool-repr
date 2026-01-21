import logging
from typing import Literal

import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


@app.command(help="Fit a feed forward network to a boolean function")
def train_ffn(
    function_class: Literal[
        "conjunction",
        "disjunction",
        "parity",
        "majority",
    ] = "conjunction",
    input_dim: int = 8,
    epochs: int = 25,
    batch_size: int = 128,
    parity_relevant_vars: int = 2,
    hidden_dim: int = 64,
    train_data_proportion: float = 0.8,
    out_dir: str = "out/ffn",
    random_seed: int | None = None,
):
    from boolrepr.scripts.train_ffn import main

    main(
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


@app.command(help="Fit a transformer model to a boolean function")
def train_transformer(
    function_class: Literal[
        "conjunction",
        "disjunction",
        "parity",
        "majority",
    ] = "conjunction",
    input_dim: int = 8,
    epochs: int = 25,
    batch_size: int = 128,
    parity_relevant_vars: int = 2,
    num_blocks: int = 3,
    num_heads: int = 2,
    hidden_dim: int = 64,
    train_data_proportion: float = 0.8,
    out_dir: str = "out/transformer",
    random_seed: int | None = None,
):
    from pathlib import Path

    from boolrepr.clustering import Clustering
    from boolrepr.data import BooleanFunctionDataset
    from boolrepr.models import TransformerEncoder
    from boolrepr.trainer import Trainer

    bool_function = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        parity_relevant_vars=parity_relevant_vars,
        random_seed=random_seed,
        transformer=True,
    )

    logger.info("dataset size %d", len(bool_function))
    logger.info(f"relevant variables {bool_function.relevant_vars}")

    model = TransformerEncoder(
        embed_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_classes=1,
    )

    trainer = Trainer(
        model=model,
        bool_function=bool_function,
        epochs=epochs,
        batch_size=batch_size,
        train_data_proportion=train_data_proportion,
        out_dir=Path(out_dir),
    )

    trainer.train()
    testing_epochs = list(range(1, epochs + 1, max(epochs // 50, 1)))
    cluster = Clustering(
        model,
        out_dir,
        testing_epochs,
        trainer.eval_loader,
        trainer.fourier_coefs,
        bool_function.relevant_vars,
    )
    cluster.test_ood(model)
    cluster.correlate()
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
        "figure_transformer.pdf",
    )
    logger.info("Clustering info saved to figure_transformer.pdf")


if __name__ == "__main__":
    app()
