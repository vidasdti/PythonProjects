import numpy as np


def check_winner(state):
    """
    Check whether a player has completed a winning line.
    Returns:
        1  -> X wins
       -1  -> O wins
        0  -> No winner
    """
    wins = [
        [0,1,2],[3,4,5],[6,7,8],
        [0,3,6],[1,4,7],[2,5,8],
        [0,4,8],[2,4,6]
    ]

    for a,b,c in wins:
        if state[a] == state[b] == state[c] and state[a] != 0:
            return state[a]

    return 0


class TicTacToeEnv:
    """
    TicTacToe environment for DQN training and evaluation.
    """

    def __init__(self):
        self.reset()

    def check_winner(self):
        return check_winner(self.board.ravel())

    def reset(self):
        """Reset the board and start a new game."""        
        self.board = np.zeros((3, 3), dtype=np.int8)
        return self.get_state()

    def get_state(self):
        """Return the flattened board state."""
        return self.board.ravel().astype(np.float32)

    def available_actions(self):
        """Return all currently legal actions."""
        board = self.board.ravel()
        return [i for i, v in enumerate(board) if v == 0]

    def step(self, action, player=1):
        """
            Execute one action in the environment.
            Returns:next_state, reward, done
        """

        row = action // 3
        col = action % 3

        # Invalid move immediately ends the episode with a penalty.
        if self.board[row, col] != 0:
            return self.get_state(), -1.0, True

        self.board[row, col] = player

        winner = self.check_winner()

        if winner == player:
            return self.get_state(), 1.0, True
        
        # Draw: board is full and no winner exists.
        if not np.any(self.board == 0):
            return self.get_state(), 0.0, True

        return self.get_state(), 0.0, False

    def render(self):
        symbols = {1: "X", -1: "O", 0: "."}

        print()
        for row in self.board:
            print(" ".join(symbols[int(c)] for c in row))
        print()