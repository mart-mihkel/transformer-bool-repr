import logging
from typing import Literal

import typer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("boolrepr")

app = typer.Typer()


@app.command()
def data(
    function_class: Literal[
        "conjunction", "disjunction", "parity", "majority"
    ] = "conjunction",
    input_dim: int = 8,
    seq_length: int = 32,
    random_seed: int | None = None,
):
    from boolrepr.data import BooleanFunctionDataset, data_visualizer

    dataset = BooleanFunctionDataset(
        num_samples=2,
        seq_length=seq_length,
        input_dim=input_dim,
        function_class=function_class,
        seed=random_seed,
    )

    data_visualizer(function_class, input_dim, seq_length, dataset)


if __name__ == "__main__":
    app()
