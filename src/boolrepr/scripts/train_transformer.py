import logging
from pathlib import Path
from typing import Literal

from boolrepr.clustering import Clustering
from boolrepr.data import BooleanFunctionDataset
from boolrepr.models import FeedForwardNetwork, TransformerEncoder
from boolrepr.trainer import Trainer

logger = logging.getLogger(__name__)


def main(
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
) -> tuple[Trainer, BooleanFunctionDataset, FeedForwardNetwork]:
    func = BooleanFunctionDataset(
        input_dim=input_dim,
        function_class=function_class,
        parity_relevant_vars=parity_relevant_vars,
        random_seed=random_seed,
        transformer=True,
    )

    logger.info("relevant variables %d", func.relevant_vars)
    logger.info("dataset size       %d", len(func))

    model = TransformerEncoder(
        embed_dim=input_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        num_classes=1,
    )

    trainer = Trainer(
        model=model,
        bool_function=func,
        epochs=epochs,
        batch_size=batch_size,
        train_data_proportion=train_data_proportion,
        out_dir=Path(out_dir),
    )

    trainer.train()

    return trainer, func, model
