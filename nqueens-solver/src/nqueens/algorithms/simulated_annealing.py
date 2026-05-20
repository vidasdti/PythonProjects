import math
import random

from src.nqueens.board import NQueensBoard


def random_neighbor(state):

    new_state = state[:]

    n = len(state)

    col = random.randint(0, n - 1)

    row = random.randint(0, n - 1)

    new_state[col] = row

    return new_state


def simulated_annealing(
    n: int,
    temperature: float = 100,
    cooling_rate: float = 0.99,
    max_steps: int = 5000
):

    board = NQueensBoard(n)

    current_state = board.state

    current_conflicts = (
        NQueensBoard.count_conflicts(current_state)
    )

    history = [current_conflicts]

    for step in range(max_steps):

        if current_conflicts == 0:

            return (
                current_state,
                0,
                step,
                history
            )

        neighbor = random_neighbor(current_state)

        neighbor_conflicts = (
            NQueensBoard.count_conflicts(neighbor)
        )

        delta = (
            current_conflicts - neighbor_conflicts
        )

        if delta > 0:

            current_state = neighbor
            current_conflicts = neighbor_conflicts

        else:

            probability = math.exp(
                delta / temperature
            )

            if random.random() < probability:

                current_state = neighbor
                current_conflicts = neighbor_conflicts

        history.append(current_conflicts)

        temperature *= cooling_rate

        if temperature < 0.001:
            break

    return (
        current_state,
        0,
        step,
        history
    )