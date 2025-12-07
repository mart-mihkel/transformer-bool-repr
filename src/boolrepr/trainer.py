import json
import logging
import os
from pathlib import Path
from typing import TypedDict

import torch
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


class Trainer:
    def __init__(
        self,
        model: Module,
        train_data: BooleanFunctionDataset,
        eval_data: BooleanFunctionDataset,
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

        self.train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        self.eval_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
        )

        self.optimizer = AdamW(model.parameters())

    def train(self):
        for epoch in range(1, self.epochs + 1):
            logger.info("epoch %d/%d", epoch, self.epochs)
            self._epoch(epoch=epoch)
            self._save_checkpoint(epoch=epoch)
            self._save_telemetry()

    def _epoch(self, epoch: int):
        train_loss = 0
        eval_loss = 0

        self.model.train()
        for batch in tqdm(self.train_loader, desc="Train"):
            x, y = batch
            y_hat = self.model(x.to(torch.device))
            train_loss += self.compute_loss(y, y_hat)
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model.eval()
        for batch in tqdm(self.eval_loader, desc="Eval"):
            x, y = batch

            with torch.no_grad():
                y_hat = self.model(x.to(torch.device))

            eval_loss += self.compute_loss(y, y_hat)

        avg_train_loss = train_loss / len(self.train_loader)
        avg_eval_loss = eval_loss / len(self.eval_loader)

        logger.info("train loss: %.4f", avg_train_loss)
        logger.info("eval loss:  %.4f", avg_eval_loss)

        self.telemetry.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "eval_loss": avg_eval_loss,
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

    @staticmethod
    def compute_loss(_label, _pred):
        raise NotImplementedError()
