import json
import logging
import os
from pathlib import Path
from typing import Annotated, TypedDict

import torch
from torch import Tensor
from torch.nn.modules.module import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from boolrepr.data import BooleanFunctionDataset

logger = logging.getLogger("boolrepr")


class Telemetry(TypedDict):
    epoch: int
    train_loss: float


class Batch(TypedDict):
    x: Annotated[Tensor, "batch input"]
    y: Annotated[Tensor, "input"]


class Trainer:
    def __init__(
        self,
        model: Module,
        bool_function: BooleanFunctionDataset,
        epochs: int = 10,
        batch_size: int = 16,
        out_dir: Path = Path("out/train"),
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.telemetry: list[Telemetry] = []

        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters())

        self.data_loader = DataLoader(
            bool_function,
            batch_size=batch_size,
            shuffle=True,
        )

    def train(self):
        for epoch in range(1, self.epochs + 1):
            logger.info("epoch %d/%d", epoch, self.epochs)
            self._epoch(epoch=epoch)
            self._save_checkpoint(epoch=epoch)
            self._save_telemetry()

    def _epoch(self, epoch: int):
        self.model.train()

        train_loss = 0
        for batch in tqdm(self.data_loader, desc="Train"):
            batch: Batch

            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            out = self.model(x).flatten()
            y_hat = torch.nn.functional.sigmoid(out)

            loss = torch.nn.functional.cross_entropy(y_hat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(self.data_loader)
        logger.info("train loss: %.4f", avg_train_loss)
        self.telemetry.append({"epoch": epoch, "train_loss": avg_train_loss})

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.out_dir, exist_ok=True)
        save_path = self.out_dir / f"chkpt-{epoch}.pth"
        logger.info("save model state dict to %s", save_path)
        torch.save(self.model.state_dict(), save_path)

    def _save_telemetry(self):
        telemetry_path = self.out_dir / "telemetry.json"
        logger.info("save model telemetry to %s", telemetry_path)
        with open(telemetry_path, "w") as f:
            json.dump(self.telemetry, f)

