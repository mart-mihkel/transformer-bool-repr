import logging
from typing import Literal

import typer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boolrepr")

app = typer.Typer()


@app.command()
def train(
    function_class: Literal[
        "conjunction", "disjunction", "parity", "majority"
    ] = "conjunction",
    input_dim: int = 8,
    seq_length: int = 32,
    seed: int = 42,
):
    from boolrepr.data import BooleanFunctionDataset

    train_dataset = BooleanFunctionDataset(
        num_samples=2,
        seq_length=seq_length,
        input_dim=input_dim,
        function_class=function_class,
        seed=seed,
    )

    val_dataset = BooleanFunctionDataset(
        num_samples=2,
        seq_length=seq_length,
        input_dim=input_dim,
        function_class=function_class,
        seed=seed + 1,
    )

    logger.info("%s", train_dataset)
    logger.info("%s", val_dataset)


if __name__ == "__main__":
    app()
