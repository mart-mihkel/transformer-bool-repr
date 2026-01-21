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
    from boolrepr.scripts.train_transformer import main

    main(
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


if __name__ == "__main__":
    app()
