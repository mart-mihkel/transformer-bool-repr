import logging
from typing import Literal
import torch

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
    epochs: int = 25,
    batch_size: int = 128,
    k: int = 2,
    out_dir: str = "out/train",
    random_seed: int | None = None,
):
    from pathlib import Path

    from boolrepr.data import BooleanFunctionDataset
    from boolrepr.models import FeedForwardNetwork
    from boolrepr.trainer import Trainer

    bool_function = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        k = k,
        random_seed=random_seed,
    )

    logger.info(f"Dataset size: {len(bool_function.data)}")   

    ffn = FeedForwardNetwork(
        input_size=input_dim,
        hidden_size=8,
        out_size=1,
    )

    trainer = Trainer(
        model=ffn,
        bool_function=bool_function,
        epochs=epochs,
        batch_size=batch_size,
        out_dir=Path(out_dir),
    )

    trainer.train() 

if __name__ == "__main__":
    app()
