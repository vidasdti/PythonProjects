import argparse
import time

from src.nqueens.board import NQueensBoard

from src.nqueens.visualization import (
    plot_board,
    plot_history
)

from src.nqueens.algorithms.hill_climbing import (
    hill_climbing
)

from src.nqueens.algorithms.simulated_annealing import (
    simulated_annealing
)


def main():

    parser = argparse.ArgumentParser(
        description="N-Queens Solver"
    )

    parser.add_argument(
        "--n",
        type=int,
        default=8,
        help="Board size"
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="hill",
        choices=["hill", "annealing"],
        help="Algorithm choice"
    )

    args = parser.parse_args()

    start_time = time.time()

    if args.algorithm == "hill":

        state, restarts, steps, history = (
            hill_climbing(args.n)
        )

    else:

        state, restarts, steps, history = (
            simulated_annealing(args.n)
        )

    runtime = time.time() - start_time

    print("\nAlgorithm:", args.algorithm)

    print("Board Size:", args.n)

    print("\nSolution:")

    print(state)

    print(
        f"\nConflicts: "
        f"{NQueensBoard.count_conflicts(state)}"
    )

    print(f"Restarts: {restarts}")

    print(f"Steps: {steps}")

    print(f"Runtime: {runtime:.4f} seconds")

    plot_board(state)

    plot_history(history)


if __name__ == "__main__":
    main()