import random
import torch

from environment import TicTacToeEnv
from dqn import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state(state, player):
    """
    Convert the board state to the player's perspective.
    """
    state = state.tolist()
    if player == 1:
        return state
    
    return [-x for x in state]


class Evaluator:
    """
    Evaluate a trained DQN agent against a random opponent.
    """

    def __init__(self):

        self.env = TicTacToeEnv()
        self.model = DQN().to(DEVICE)
        self.model.load_state_dict(torch.load("models2/player1/best_model.pth", map_location=DEVICE))
        self.model.eval()

    def choose_action(self, raw_state):

        state = normalize_state(raw_state, 1)
        valid = [i for i, v in enumerate(raw_state) if v == 0]

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.model(s)[0]

        # Mask invalid actions before selecting argmax.
        masked = torch.full((9,), float("-inf"), device=DEVICE)

        for a in valid:
            masked[a] = q[a]

        return torch.argmax(masked).item()

    def opponent(self, raw_state):
        """
        Random opponent used as a baseline for evaluation.
        """
        valid = [i for i, v in enumerate(raw_state) if v == 0]
        return random.choice(valid)


    def play_vs_random(self, games=1000):
        """
        Play multiple games against a random agent and
        report win/loss/draw statistics.
        """
        wins = 0
        losses = 0
        draws = 0

        for _ in range(games):

            state = self.env.reset()
            done = False

            while not done:

                action = self.choose_action(state)
                state, _, done = self.env.step(action, player=1)

                if done:
                    break

                opp_action = self.opponent(state)
                state, _, done = self.env.step(opp_action, player=-1)

            winner = self.env.check_winner()

            if winner == 1:
                wins += 1
            elif winner == -1:
                losses += 1
            else:
                draws += 1

        print("\n===== RESULTS =====")
        print(f"Wins   : {wins}")
        print(f"Losses : {losses}")
        print(f"Draws  : {draws}")
        print(f"Win Rate: " f"{100 * wins / games:.2f}%")


if __name__ == "__main__":

    evaluator = Evaluator()
    evaluator.play_vs_random(games=10000)