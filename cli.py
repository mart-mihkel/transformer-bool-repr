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
    epochs: int = 25,
    model_type: str = "ffn",  # it seems typer doesn't support Literals
    batch_size: int = 128,
    k: int = 2,
    num_transformer_blocks: int = 3,
    num_transformer_heads: int = 2,
    hidden_dim: int = 64,
    out_dir: str = "out/train",
    random_seed: int | None = None,
):
    from pathlib import Path

    from boolrepr.data import BooleanFunctionDataset
    from boolrepr.models import FeedForwardNetwork, TransformerEncoder
    from boolrepr.trainer import Trainer

    if model_type == "transformer":
        bool_function = BooleanFunctionDataset(
            input_dim=input_dim,
            function_class=function_class,
            k=k,
            random_seed=random_seed,
            transformer=True,
        )

        model = TransformerEncoder(
            embed_dim=input_dim,
            num_heads=num_transformer_heads,
            hidden_dim=hidden_dim,
            num_blocks=num_transformer_blocks,
            num_classes=1,
        )
    else:
        bool_function = BooleanFunctionDataset(
            input_dim=input_dim,
            function_class=function_class,
            k=k,
            random_seed=random_seed,
        )

        model = FeedForwardNetwork(
            input_size=input_dim,
            hidden_size=hidden_dim,
            out_size=1,
        )

    logger.info(f"Dataset size: {len(bool_function.data)}")

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
