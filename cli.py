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
    epochs: int = 25,
    batch_size: int = 32,
    out_dir: str = "out/train",
    random_seed: int | None = None,
):
    from pathlib import Path

    from boolrepr.data import BooleanFunctionDataset
    from boolrepr.models import FeedForwardNetwork
    from boolrepr.trainer import Trainer

    train_data = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        seed=random_seed,
    )

    eval_data = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        seed=random_seed,
    )

    ffn = FeedForwardNetwork(
        input_size=input_dim * seq_length,
        hidden_size=256,
        out_size=seq_length,
    )

    trainer = Trainer(
        model=ffn,
        train_data=train_data,
        eval_data=eval_data,
        epochs=epochs,
        batch_size=batch_size,
        out_dir=Path(out_dir),
        collate_fn=BooleanFunctionDataset.collate_fn_feed_forward,
    )

    trainer.train()


if __name__ == "__main__":
    app()
