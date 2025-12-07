import random
import logging
from typing import Callable, Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


logger = logging.getLogger("boolrepr")

type FunctionClass = Literal["conjunction", "disjunction", "parity", "majority"]


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
        seed: int | None = 42,
    ):
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

    def _generate_function(self) -> Callable[[Tensor], Tensor]:
        """Generate a function from the specified class."""
        if self.function_class == "conjunction":
            return self._generate_conjunction()
        elif self.function_class == "disjunction":
            return self._generate_disjunction()
        elif self.function_class == "parity":
            return self._generate_parity()
        elif self.function_class == "majority":
            return self._generate_majority()
        else:
            raise ValueError(f"Unknown function class: {self.function_class}")

    def _generate_conjunction(self) -> Callable[[Tensor], Tensor]:
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

        def conjunction_func(x: Tensor) -> Tensor:
            """Evaluate the conjunction on input x."""
            result = torch.ones(x.shape[0], dtype=torch.float32)
            for var_idx, is_positive in literals:
                if is_positive:
                    result = result * x[:, var_idx]
                else:
                    result = result * (1 - x[:, var_idx])
            return result

        return conjunction_func

    def _generate_parity(self, k: int = 2) -> Callable[[Tensor], Tensor]:
        """
        Generate a parity function on k random variables.
        For PARITY-(n,k).
        """
        # Randomly select k variables
        relevant_vars = random.sample(range(self.input_dim), k)

        def parity_func(x: Tensor) -> Tensor:
            """Compute XOR of relevant variables."""
            # Take only relevant variables
            relevant = x[:, relevant_vars]
            # Compute XOR: sum mod 2
            parity = torch.sum(relevant, dim=1) % 2
            return parity.float()

        return parity_func

    def _generate_disjunction(self) -> Callable[[Tensor], Tensor]:
        """Generate a random disjunction function."""
        # Similar to conjunction but with OR instead of AND
        literals = []
        for i in range(self.input_dim):
            r = random.random()
            if r < 0.15:
                literals.append((i, 1))
            elif r < 0.30:
                literals.append((i, 0))

        def disjunction_func(x: Tensor) -> Tensor:
            result = torch.zeros(x.shape[0], dtype=torch.float32)
            for var_idx, is_positive in literals:
                if is_positive:
                    result = result + x[:, var_idx]
                else:
                    result = result + (1 - x[:, var_idx])
            # OR operation: if any literal is true, result > 0
            return (result > 0).float()

        return disjunction_func

    def _generate_majority(self) -> Callable[[Tensor], Tensor]:
        """Generate a majority function on a random subset of variables."""
        # Random subset of variables (size ~n/3)
        subset_size = max(1, self.input_dim // 3)
        relevant_vars = random.sample(range(self.input_dim), subset_size)

        def majority_func(x: Tensor) -> Tensor:
            relevant = x[:, relevant_vars]
            # Majority: if more than half are 1, output 1
            majority_threshold = relevant.shape[1] / 2
            return (torch.sum(relevant, dim=1) > majority_threshold).float()

        return majority_func

    def _generate_inputs(self, num_inputs: int) -> Tensor:
        """Generate random Boolean inputs."""
        # The paper uses a modified distribution for some tasks.
        # We just use uniform for now.
        return torch.bernoulli(0.5 * torch.ones(num_inputs, self.input_dim))

    def _create_teaching_sequence(self) -> Tensor:
        """Create a teaching sequence for the given function."""
        # TODO: Placeholder for now
        return torch.empty(0)

    def _generate_sequence(self) -> Tensor:
        """Generate one complete sequence with unified dimensions."""
        func = self._generate_function()

        # Generate inputs
        all_inputs = self._generate_inputs(self.seq_length)

        # Get labels
        labels = func(all_inputs)

        # Add noise
        if self.noise_prob > 0:
            noise_mask = torch.bernoulli(self.noise_prob * torch.ones_like(labels))
            labels = (labels + noise_mask) % 2

        # 5. Format and Interleave
        # x_token: [x_1, ... x_n, 0]
        # y_token: [0, ... 0, y]

        # Prepare x tokens: Pad with one zero at the end
        zeros_for_x = torch.zeros(self.seq_length, 1)
        x_tokens = torch.cat([all_inputs, zeros_for_x], dim=1)

        # Prepare y tokens: Pad with zeros at the start, label at the end
        zeros_for_y = torch.zeros(self.seq_length, self.input_dim)
        y_tokens = torch.cat([zeros_for_y, labels.unsqueeze(1)], dim=1)

        # Interleave
        # Stack inputs and labels alternately
        sequence = torch.empty(self.seq_length * 2, self.input_dim + 1)
        sequence[0::2] = x_tokens
        sequence[1::2] = y_tokens

        return sequence

    def _generate_data(self) -> Tensor:
        """Generate all sequences."""
        data = torch.empty(self.num_samples, self.seq_length * 2, self.input_dim + 1)
        for i in range(self.num_samples):
            data[i] = self._generate_sequence()

        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


def data_visualizer(args, dataset):
    logger.info(f"\nDataset type: {type(dataset)}")
    logger.info(f"Dataset length: {len(dataset)}")
    logger.info(f"Function class: {args.function}")
    logger.info(f"Input dimension: {args.input_dim}")
    logger.info(f"Sequence length: {args.seq_length}")

    # Show first sequence in detail
    logger.info(f"\n{'=' * 80}")
    logger.info(f"FIRST SEQUENCE ANALYSIS (function: {args.function}):")
    logger.info(f"{'=' * 80}")

    seq = dataset[0]
    logger.info(f"\nFull sequence shape: {seq.shape}")
    logger.info(
        f"Expected shape: (2*seq_length, input_dim+1) = ({2 * args.seq_length}, {args.input_dim + 1})"
    )

    # Show the raw tensor (first few rows)
    logger.info("\nFirst 6 rows of the raw tensor:")
    for i in range(min(6, len(seq))):
        logger.info(f"  Row {i:2d}: {seq[i].int().tolist()}")

    # Decode the sequence
    logger.info(f"\n{'=' * 80}")
    logger.info("DECODED (x,y) PAIRS (first 5 examples):")
    logger.info(f"{'=' * 80}")

    for i in range(min(5, args.seq_length)):
        x_vec = seq[2 * i]
        y_vec = seq[2 * i + 1]

        x_input = x_vec[:-1]  # Actual input bits
        x_padding = x_vec[-1]  # Last element is padding (should be 0)

        y_padding = y_vec[:-1]  # First elements are padding (should be all 0s)
        y_label = y_vec[-1]  # Last element is the actual label

        logger.info(f"\nExample {i}:")
        logger.info(f"  x_{i}: {x_input.int().tolist()}")
        logger.info(f"  x padding indicator: {x_padding.item()}")
        logger.info(f"  y_{i}: {y_label.item():.0f}")
        logger.info(f"  y padding: {y_padding.int().tolist()}")

        binary = "".join(str(int(b)) for b in x_input)
        grouped_binary = " ".join([binary[j : j + 4] for j in range(0, len(binary), 4)])
        logger.info(f"  Binary: {grouped_binary}")

    logger.info(f"\n{'=' * 80}")
    logger.info("LABEL STATISTICS FOR FIRST SEQUENCE:")
    logger.info(f"{'=' * 80}")

    labels = []
    for i in range(args.seq_length):
        y_vec = seq[2 * i + 1]
        y_label = y_vec[-1]
        labels.append(int(y_label.item()))

    logger.info(f"Labels: {labels}")
    logger.info(f"Number of 1s: {sum(labels)}")
    logger.info(f"Number of 0s: {len(labels) - sum(labels)}")
    logger.info(f"Proportion of 1s: {sum(labels) / len(labels):.2f}")

    logger.info(f"\n{'=' * 80}")
    logger.info("CHECKING MULTIPLE SEQUENCES (first label from each):")
    logger.info(f"{'=' * 80}")

    for seq_idx in range(min(3, len(dataset))):
        seq = dataset[seq_idx]
        y_vec = seq[1]
        y_label = y_vec[-1]
        logger.info(f"Sequence {seq_idx}: first label = {int(y_label.item())}")

        x0 = seq[0][:-1]
        binary = "".join(str(int(b)) for b in x0)
        grouped_binary = " ".join([binary[j : j + 4] for j in range(0, len(binary), 4)])
        logger.info(f"            first input = {grouped_binary}")
