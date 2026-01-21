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
from itertools import combinations

from boolrepr.data import BooleanFunctionDataset

logger = logging.getLogger("boolrepr")


class Telemetry(TypedDict):
    epoch: int
    train_loss: float
    train_accuracy: float
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
        self.fourier_coefs = None
        # self.fourier_expand(bool_function)

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
        train_acc = 0
        pbar = tqdm(self.train_loader, desc="Train")
        for i, batch in enumerate(pbar):
            batch: Batch

            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            y_hat = self.model(x)[0].flatten()

            # TESTING (-1,1) to (1, 0)
            # y_hat = ((-1) * y_hat + 1)/2

            y = ((-1) * y + 1) / 2
            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

            # loss = torch.nn.functional.mse_loss(y_hat, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss.item()
            train_acc += ((y_hat > 0.5) == y.bool()).float().mean().item()
            # train_acc += ((y_hat < 0) == (y < 0)).float().mean().item()
            pbar.set_description(f"Train loss {train_loss / (i + 1):.4f}")
        self._eval(epoch, train_loss, train_acc)

    def _eval(self, epoch: int, train_loss: float, train_acc: float):
        self.model.eval()
        eval_acc = 0
        eval_loss = 0
        pbar = tqdm(self.eval_loader, desc="Eval")
        for i, batch in enumerate(pbar):
            batch: Batch

            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            with torch.no_grad():
                y_hat = self.model(x)[0].flatten()

            # Testing (0,1) to (1, -1)
            # y_hat = ((-1) * y_hat + 1)/2

            y = ((-1) * y + 1) / 2
            loss = torch.nn.functional.binary_cross_entropy(y_hat, y)

            # loss = torch.nn.functional.mse_loss(y_hat, y)

            eval_loss += loss.item()
            eval_acc += ((y_hat > 0.5) == y.bool()).float().mean().item()
            # eval_acc += ((y_hat < 0) == (y < 0)).float().mean().item()
            pbar.set_description(f"Eval loss {eval_loss / (i + 1):.4f}")

        avg_train_loss = train_loss / len(self.train_loader)
        avg_train_acc = train_acc / len(self.train_loader)
        avg_eval_loss = eval_loss / len(self.eval_loader)
        avg_eval_acc = eval_acc / len(self.eval_loader)

        logger.info("train loss:    %.4f", avg_train_loss)
        logger.info("train accuracy:    %.4f", avg_train_acc)
        logger.info("eval loss:     %.4f", avg_eval_loss)
        logger.info("eval accuracy: %.4f", avg_eval_acc)

        self.telemetry.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_accuracy": avg_train_acc,
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
        os.makedirs(self.out_dir, exist_ok=True)
        telemetry_path = self.out_dir / "telemetry.json"
        logger.info("save telemetry to %s", telemetry_path)
        with open(telemetry_path, "w") as f:
            json.dump(self.telemetry, f)

    def fourier_expand(self, dataset):
        truth_table = dataset.data
        if len(truth_table[0]["x"]) == 1:
            input_dim = len(truth_table[0]["x"][0])
        else:
            input_dim = len(truth_table[0]["x"])
        assert 2**input_dim == len(truth_table), (
            "Truth table size does not match input dimension"
        )
        input_indices = list(range(0, input_dim))
        term_coefs = []
        for size_of_term in range(0, input_dim + 1):
            if size_of_term > len(dataset.relevant_vars):
                break
            term_indices = list(combinations(input_indices, size_of_term))
            print(f"Terms of size {size_of_term}: {len(term_indices)}")
            for index_combination in term_indices:
                term_value = 0
                for row in truth_table:
                    temp_product = row["y"].item()
                    for index in index_combination:
                        if len(row["x"]) == 1:
                            temp_product *= row["x"][0][index].item()
                        else:
                            temp_product *= row["x"][index].item()
                    term_value += temp_product
                term_value = (1 / (2**input_dim)) * term_value
                if term_value != 0:
                    print(f"Term {index_combination} has value {term_value}")
                """
                term_value = 0
                for row in truth_table:
                    temp_product = (-1)**row["y"].item()
                    for index in index_comb:
                        if len(row["x"]) == 1:
                            temp_product *= (-1)**row["x"][0][index].item()
                        else:
                            temp_product *= (-1)**row["x"][index].item()
                    term_value += temp_product
                if term_value != 0:
                    print(f"Term {index_comb} has value {(1/(2**input_dim)) * term_value}")
                """
                term_coefs.append((index_combination, term_value))

        logger.info(
            "Coefficients with non-zero values %d",
            sum([1 for term in term_coefs if term[1] != 0]),
        )
        self.fourier_coefs = term_coefs
