import logging
import random
from typing import Annotated, Literal, TypedDict

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger("boolrepr")

type FunctionClass = Literal["conjunction", "disjunction", "parity", "majority"]


class FunctionSequence(TypedDict):
    x: Annotated[Tensor, "sequence input"]
    y: Annotated[Tensor, "sequence"]


class BooleanFunctionDataset(Dataset):
    """
    Generates sequences for in-context learning of Boolean functions.
    Each sequence is: (x₁, f(x₁), x₂, f(x₂), ..., xₘ, f(xₘ))
    """

    def __init__(
        self,
        num_samples: int = 10000,
        seq_length: int = 70,
        input_dim: int = 28,
        function_class: FunctionClass = "conjunction",
        noise_prob: float = 0.0,
        teaching_sequence: bool = False,
        seed: int | None = None,
    ):
        assert function_class in ["conjunction", "disjunction", "parity", "majority"], (
            "Invalid function class"
        )

        self.num_samples = num_samples
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.function_class = function_class
        self.noise_prob = noise_prob
        self.teaching_sequence = teaching_sequence

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data = self._generate_data()

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

    def _generate_parity(self, x: Tensor, k: int = 2) -> Tensor:
        """
        Generate a parity function on k random variables.
        For PARITY-(n,k).
        """
        # Randomly select k variables
        relevant_vars = random.sample(range(self.input_dim), k)

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
        """Generate random Boolean inputs."""
        # The paper uses a modified distribution for some tasks.
        # We just use uniform for now.
        return torch.bernoulli(0.5 * torch.ones(self.seq_length, self.input_dim))

    def _create_teaching_sequence(self) -> Tensor:
        """Create a teaching sequence for the given function."""
        # TODO: Placeholder for now
        return torch.empty(0)

    def _generate_sequence(self) -> FunctionSequence:
        """Generate one complete sequence with unified dimensions."""
        inputs = self._generate_inputs()
        labels = self._generate_labels(inputs)

        if self.noise_prob > 0:
            noise_mask = torch.bernoulli(self.noise_prob * torch.ones_like(labels))
            labels = (labels + noise_mask) % 2

        return {"x": inputs, "y": labels}

    def _generate_data(self) -> list[FunctionSequence]:
        """Generate all sequences."""
        return [self._generate_sequence() for _ in range(self.num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn_feed_forward(batch: list[FunctionSequence]) -> dict[str, Tensor]:
        x = [seq["x"].flatten() for seq in batch]
        y = [seq["y"] for seq in batch]
        return {"x": torch.stack(x), "y": torch.stack(y)}
