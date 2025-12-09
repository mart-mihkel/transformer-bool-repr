import logging
from typing import Literal

import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boolrepr")

app = typer.Typer()


@app.command()
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
    out_dir: str = "out/ffn",
    random_seed: int | None = None,
):
    from pathlib import Path

    from boolrepr.data import BooleanFunctionDataset
    from boolrepr.models import FeedForwardNetwork
    from boolrepr.trainer import Trainer

    bool_function = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        parity_relevant_vars=parity_relevant_vars,
        random_seed=random_seed,
    )

    logger.info("dataset size %d", len(bool_function))

    model = FeedForwardNetwork(
        input_size=input_dim,
        hidden_size=hidden_dim,
        out_size=1,
    )

    trainer = Trainer(
        model=model,
        bool_function=bool_function,
        epochs=epochs,
        batch_size=batch_size,
        out_dir=Path(out_dir),
    )

    trainer.train()


@app.command()
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
    out_dir: str = "out/transformer",
    random_seed: int | None = None,
):
    from pathlib import Path

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
        out_dir=Path(out_dir),
    )

    trainer.train()


if __name__ == "__main__":
    app()
