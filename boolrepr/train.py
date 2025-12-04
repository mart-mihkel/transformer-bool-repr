from data import BooleanFunctionDataset, data_visualizer
import argparse

"""
    To run, E.g.: uv run train.py --function conjunction --input_dim 8 --seq_length 6
"""


def main():
    parser = argparse.ArgumentParser(description="Train in-context learning model")
    parser.add_argument(
        "--function",
        type=str,
        default="conjunction",
        choices=["conjunction", "disjunction", "parity", "majority"],
    )
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Create datasets
    train_dataset = BooleanFunctionDataset(
        num_samples=3,  # Few for debugging
        seq_length=args.seq_length,
        input_dim=args.input_dim,
        function_class=args.function,
        seed=args.seed,
    )

    val_dataset = BooleanFunctionDataset(
        num_samples=2,
        seq_length=args.seq_length,
        input_dim=args.input_dim,
        function_class=args.function,
        seed=args.seed + 1,
    )

    # DEBUG: Visualize data
    # data_visualizer(args, train_dataset)

    # TODO: model training.


if __name__ == "__main__":
    main()
