import math
from functools import lru_cache

from environment import check_winner


def valid_moves(state):
    """Return all empty positions on the board."""
    return [i for i, v in enumerate(state) if v == 0]


@lru_cache(maxsize=None)
def minimax(state_tuple, player):
    """
    Compute the minimax value of a board state.

    Returns:
        1  -> player X wins
        0  -> draw
        -1 -> player O wins
    """

    state = list(state_tuple)

    winner = check_winner(state)

    if winner == 1:
        return 1

    if winner == -1:
        return -1

    moves = valid_moves(state)

    if not moves:
        return 0

    if player == 1:

        best = -math.inf

        for move in moves:

            next_state = state.copy()
            next_state[move] = 1

            best = max(
                best,
                minimax(tuple(next_state), -1)
            )

        return best

    best = math.inf

    for move in moves:

        next_state = state.copy()
        next_state[move] = -1

        best = min(
            best,
            minimax(tuple(next_state), 1)
        )

    return best


def minimax_move(state, as_player):
    """
    Return the optimal move for the specified player.
    """

    moves = valid_moves(state)

    best_score = -math.inf
    best_move = None

    for move in moves:

        next_state = state.copy()
        next_state[move] = as_player

        score = minimax(
            tuple(next_state),
            -as_player
        )

        if as_player == -1:
            score = -score

        if score > best_score:

            best_score = score
            best_move = move

    return best_move