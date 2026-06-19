import torch

from environment import TicTacToeEnv
from dqn import DQN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state(state, player):
    """
    Convert the board state to the current player's perspective.
    Player -1 sees the board with inverted values.
    """
    state = state.tolist()
    if player == 1:
        return state
    return [-x for x in state]


def choose_action(model, env, player):

    state = normalize_state(env.get_state(), player)
    valid_actions = env.available_actions()

    with torch.no_grad():

        s = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        q_values = model(s)[0]
        
    # Mask invalid actions before selecting argmax.
    masked = torch.full((9,), float("-inf"), device=DEVICE)

    for a in valid_actions:
        masked[a] = q_values[a] 
    
    return torch.argmax(masked).item()


def load_model(path):

    model = DQN().to(DEVICE)
    model.load_state_dict(torch.load( path, map_location=DEVICE, weights_only=True))
    model.eval()

    return model


def play_game(player1_model, player2_model):
    """
    Run a complete game between the two trained agents.
    Returns:
        1  -> Player 1 wins
       -1  -> Player 2 wins
        0  -> Draw
    """
    env = TicTacToeEnv()
    current_player = 1
    while True:

        if current_player == 1:
            action = choose_action(player1_model, env, 1)
        else:
            action = choose_action(player2_model, env, -1)

        _, _, done = env.step(action, player=current_player)

        if done:

            winner = env.check_winner()
            return winner
        current_player *= -1


def evaluate(num_games=1000):

    player1_model = load_model("models2/player1/best_model.pth")
    player2_model = load_model("models2/player2/best_model.pth")

    p1_wins = 0
    p2_wins = 0
    draws = 0

    for game in range(num_games):

        winner = play_game(player1_model, player2_model)

        if winner == 1:
            p1_wins += 1
        elif winner == -1:
            p2_wins += 1
        else:
            draws += 1

        if (game + 1) % 100 == 0:

            print(f"{game+1}/{num_games}")

    print("\n===== RESULTS =====")
    print(f"Player1 (X) Wins : {p1_wins}")
    print(f"Player2 (O) Wins : {p2_wins}")
    print(f"Draws            : {draws}")
    print()
    print(f"Player1 Win Rate: " f"{100*p1_wins/num_games:.2f}%")
    print(f"Player2 Win Rate: " f"{100*p2_wins/num_games:.2f}%")
    print(f"Draw Rate: " f"{100*draws/num_games:.2f}%")

if __name__ == "__main__":
    evaluate(1000)