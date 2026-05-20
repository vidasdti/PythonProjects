import random
from typing import List, Tuple

from src.nqueens.board import NQueensBoard


def find_best_neighbor(
    state: List[int]
) -> Tuple[List[int], int]:

    n = len(state)

    best_neighbors = []

    lowest_conflicts = None

    for col in range(n):

        for row in range(n):

            if state[col] == row:
                continue

            new_state = state[:]

            new_state[col] = row

            conflicts = NQueensBoard.count_conflicts(
                new_state
            )

            if (
                lowest_conflicts is None
                or conflicts < lowest_conflicts
            ):

                lowest_conflicts = conflicts

                best_neighbors = [new_state]

            elif conflicts == lowest_conflicts:

                best_neighbors.append(new_state)
    chosen = random.choice(best_neighbors)

    return chosen, lowest_conflicts


def hill_climbing(
    n: int,
    max_restarts: int = 100,
    max_steps: int = 1000
):

    for restart in range(max_restarts):

        board = NQueensBoard(n)

        state = board.state

        conflicts = NQueensBoard.count_conflicts(
            state
        )

        history = [conflicts]

        for step in range(max_steps):

            if conflicts == 0:

                return (
                    state,
                    restart,
                    step,
                    history
                )

            neighbor, neighbor_conflicts = (
                find_best_neighbor(state)
            )

            if neighbor_conflicts < conflicts:

                state = neighbor

                conflicts = neighbor_conflicts

                history.append(conflicts)

            else:
                break

    return state, restart, step, history