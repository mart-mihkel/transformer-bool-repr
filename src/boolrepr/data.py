import itertools
import logging
import random
from typing import Annotated, Literal, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger("boolrepr")

type FunctionClass = Literal["conjunction", "disjunction", "parity", "majority"]


class BooleanExpression(TypedDict):
    x: Annotated[Tensor, "input"]
    y: Annotated[Tensor, "1"]


class BooleanFunctionDataset(Dataset):
    def __init__(
        self,
        input_dim: int = 28,
        k: int = 2,
        function_class: FunctionClass = "conjunction",
        random_seed: int | None = None,
        transformer: bool = False,
    ):
        logger.info("init boolean function dataset")

        assert function_class in ["conjunction", "disjunction", "parity", "majority"], (
            "Invalid function class"
        )

        self.input_dim = input_dim
        self.seq_length = 2**input_dim
        self.function_class = function_class
        self.k = k
        self.transformer = transformer

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        self.data = self._generate_function()

    def _generate_labels(self, x: Tensor) -> Tensor:
        """Generate a function from the specified class."""
        if self.function_class == "conjunction":
            return self._generate_conjunction(x)

        if self.function_class == "disjunction":
            return self._generate_disjunction(x)

        if self.function_class == "parity":
            return self._generate_parity(x)

        return self._generate_majority(x)

    def _generate_conjunction(self, x: Tensor) -> Tensor:
        """
        Generate a random conjunction function.
        Each literal (x_i or ¬x_i) has 30% probability of being included.
        """
        # For each variable, decide: 0=exclude, 1=include positive, 2=include negative
        literals = []
        for i in range(self.input_dim):
            r = random.random()
            if r < 0.15:  # Include positive literal x_i
                literals.append((i, 1))
            elif r < 0.30:  # Include negative literal ¬x_i
                literals.append((i, 0))
            # Otherwise exclude

        # Evaluate the conjunction on input x.
        result = torch.ones(x.shape[0], dtype=torch.float32)
        for var_idx, is_positive in literals:
            if is_positive:
                result = result * x[:, var_idx]
            else:
                result = result * (1 - x[:, var_idx])

        return result

    def _generate_parity(self, x: Tensor) -> Tensor:
        """
        Generate a parity function on k random variables.
        For PARITY-(n,k).
        """
        # Randomly select k variables
        relevant_vars = random.sample(range(self.input_dim), self.k)

        # Compute XOR of relevant variables.
        # Take only relevant variables
        relevant = x[:, relevant_vars]
        # Compute XOR: sum mod 2
        parity = torch.sum(relevant, dim=1) % 2
        return parity.float()

    def _generate_disjunction(self, x: Tensor) -> Tensor:
        """Generate a random disjunction function."""
        # Similar to conjunction but with OR instead of AND
        literals = []
        for i in range(self.input_dim):
            r = random.random()
            if r < 0.15:
                literals.append((i, 1))
            elif r < 0.30:
                literals.append((i, 0))

        result = torch.zeros(x.shape[0], dtype=torch.float32)
        for var_idx, is_positive in literals:
            if is_positive:
                result = result + x[:, var_idx]
            else:
                result = result + (1 - x[:, var_idx])

        # OR operation: if any literal is true, result > 0
        return (result > 0).float()

    def _generate_majority(self, x: Tensor) -> Tensor:
        """Generate a majority function on a random subset of variables."""
        # Random subset of variables (size ~n/3)
        subset_size = max(1, self.input_dim // 3)
        relevant_vars = random.sample(range(self.input_dim), subset_size)

        relevant = x[:, relevant_vars]
        # Majority: if more than half are 1, output 1
        majority_threshold = relevant.shape[1] / 2
        return (torch.sum(relevant, dim=1) > majority_threshold).float()

    def _generate_inputs(self) -> Tensor:
        """Generate all possible Boolean input combinations."""
        # Create all combinations: [0,0,0], [0,0,1], [0,1,0], ..., [1,1,1]
        all_combinations = list(itertools.product([0, 1], repeat=self.input_dim))
        # Convert to tensor
        return torch.tensor(all_combinations, dtype=torch.float32)

    def _generate_function(self) -> list[BooleanExpression]:
        """Generate one complete sequence with all combinations."""
        inputs = self._generate_inputs()
        labels = self._generate_labels(inputs)

        # Turn variables into embeddings for the Transformer model
        if self.transformer:
            inputs = inputs.unsqueeze(1)
        return [{"x": x, "y": y} for x, y in zip(inputs, labels)]

    def __len__(self):
        return self.seq_length

    def __getitem__(self, idx):
        return self.data[idx]
