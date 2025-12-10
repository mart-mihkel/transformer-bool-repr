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
    eval_loss: float
    eval_accuracy: float


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
        train_data_proportion: float = 0.8,
        out_dir: Path = Path("out/train"),
    ) -> None:
        logger.info("init trainer")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epochs = epochs
        self.batch_size = batch_size
        self.out_dir = out_dir
        self.telemetry: list[Telemetry] = []

        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters())

        train_dataset, test_dataset = torch.utils.data.random_split(
            bool_function, [train_data_proportion, 1 - train_data_proportion]
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.eval_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
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
        pbar = tqdm(self.train_loader, desc="Train")
        for i, batch in enumerate(pbar):
            batch: Batch

            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            y_hat = self.model(x).flatten()

            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            pbar.set_description(f"Train loss {train_loss / (i + 1):.4f}")

        self.model.eval()
        eval_acc = 0
        eval_loss = 0
        pbar = tqdm(self.eval_loader, desc="Eval")
        for i, batch in enumerate(pbar):
            batch: Batch

            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            with torch.no_grad():
                y_hat = self.model(x).flatten()

            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

            eval_loss += loss.item()
            eval_acc += ((y_hat > 0.5) == y.bool()).float().mean().item()
            pbar.set_description(f"Eval loss {eval_loss / (i + 1):.4f}")

        avg_train_loss = train_loss / len(self.train_loader)
        avg_eval_loss = eval_loss / len(self.eval_loader)
        avg_eval_acc = eval_acc / len(self.eval_loader)

        logger.info("train loss:    %.4f", avg_train_loss)
        logger.info("eval loss:     %.4f", avg_eval_loss)
        logger.info("eval accuracy: %.4f", avg_eval_acc)

        self.telemetry.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
                "eval_accuracy": avg_eval_acc,
            }
        )

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
