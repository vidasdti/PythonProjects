import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy

from functools import lru_cache
from environment import TicTacToeEnv, check_winner
from dqn import DQN
from replay_buffer import ReplayBuffer
from minimax import minimax_move

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_state(state, player):
    """
    Convert the board to the current player's perspective.
    The network always sees itself as player 1.
    """
    if player == 1:
        return list(state)
    return [-x for x in state]


def valid_moves(state):
    """Return indices of all empty cells."""
    return [i for i, v in enumerate(state) if v == 0]


class SingleRoleTrainer:
    """
    Train a DQN agent that always plays a fixed role
    (X or O) throughout training.
    """

    def __init__(self, fixed_player: int, episodes: int = 5000, phase_label: str = ""):
        assert fixed_player in (1, -1), "fixed_player must be 1 or -1"

        self.fixed_player  = fixed_player
        self.opponent      = -fixed_player
        self.episodes      = episodes
        self.phase_label   = phase_label

        self.env = TicTacToeEnv()

        # ---- networks ----
        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.opponent_snapshot = copy.deepcopy(self.policy_net)
        self.opponent_snapshot.eval()

        # ---- optimiser ----
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.loss_fn   = nn.SmoothL1Loss()

        self.buffer = ReplayBuffer(capacity=50000)

        # ---- hyper-params ----
        self.gamma         = 0.99
        self.batch_size    = 64
        self.epsilon       = 1.0
        self.epsilon_min   = 0.05
        self.epsilon_decay = 0.999
        self.target_update = 500

        self.training_steps = 0
        self.best_score     = -float("inf")

        role = "player1" if fixed_player == 1 else "player2"
        os.makedirs(f"models2/{role}", exist_ok=True)
        self.save_dir = f"models2/{role}"


    def _agent_action(self, norm_state):
        """Epsilon-greedy action for the agent (normalised state)."""
        valid = valid_moves(norm_state)

        if random.random() < self.epsilon:
            return random.choice(valid)

        with torch.no_grad():
            s = torch.tensor(norm_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.policy_net(s)[0]

        masked = torch.full((9,), float("-inf"), device=DEVICE)
        for a in valid:
            masked[a] = q[a]

        return torch.argmax(masked).item()

    def _eval_action(self, norm_state):
        """Greedy action (no exploration) for evaluation."""
        valid = valid_moves(norm_state)

        with torch.no_grad():
            s = torch.tensor(norm_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.policy_net(s)[0]

        masked = torch.full((9,), float("-inf"), device=DEVICE)
        for a in valid:
            masked[a] = q[a]

        return torch.argmax(masked).item()

    def _opponent_action(self, raw_state):
        """
        Select an opponent move using a curriculum strategy.
        Early training: Mostly random opponents.
        Mid training: Mix of random, self-play policy and minimax.
        Late training: Mostly minimax opponents.
        This prevents the agent from overfitting to weak opponents
        and stabilizes learning.
        """
        valid = valid_moves(raw_state)

        # Opponent evaluates the board from its own perspective.
        opp_norm = normalize_state(raw_state, self.opponent)

        with torch.no_grad():
            s = torch.tensor(opp_norm, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.opponent_snapshot(s)[0]

        masked_q = torch.full((9,), float("-inf"), device=DEVICE)
        for a in valid:
            masked_q[a] = q[a]

        mm_move = minimax_move(raw_state, self.opponent)

        # Curriculum: gradually replace random opponents with minimax.
        p_minimax = min(0.7,self.training_steps / 50000)
        p_random  = max(0.05, 0.4 - self.training_steps / 50000)

        total = p_minimax + p_random
        if total > 0.9:
            scale     = 0.9 / total
            p_minimax *= scale
            p_random  *= scale

        p_policy = 1.0 - (p_minimax + p_random)

        r = random.random()
        if r < p_random:
            return random.choice(valid)
        elif r < p_random + p_policy:
            return torch.argmax(masked_q).item()
        else:
            return mm_move

    def _center_bonus(self, before_norm, after_norm):
        """Small bonus for taking the centre square."""
        if before_norm[4] == 0 and after_norm[4] == 1:
            return 0.1
        return 0.0

    def _two_in_row(self, norm_state):
        """
        Reward shaping based on immediate tactical threats.
        """
        lines = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6],
        ]
        score = 0
        for a, b, c in lines:
            line = [norm_state[a], norm_state[b], norm_state[c]]
            if line.count(1) == 2 and line.count(0) == 1:
                score += 1
            if line.count(-1) == 2 and line.count(0) == 1:
                score -= 5
        return score


    def _optimize(self):
        """
        Perform one Double-DQN optimization step.
        """
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states      = torch.tensor(states,      dtype=torch.float32, device=DEVICE)
        actions     = torch.tensor(actions,     dtype=torch.long,    device=DEVICE).unsqueeze(1)
        rewards     = torch.tensor(rewards,     dtype=torch.float32, device=DEVICE)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=DEVICE)
        dones       = torch.tensor(dones,       dtype=torch.float32, device=DEVICE)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():

            # Double DQN:
            # policy_net selects the next action,
            # target_net evaluates it.
            policy_q = self.policy_net(next_states)
            next_actions = []
            for i in range(next_states.size(0)):
                empty = torch.isclose(next_states[i], torch.zeros_like(next_states[i]))
                if empty.sum() == 0:
                    next_actions.append(0)
                    continue
                q = policy_q[i].clone()
                q[~empty] = float("-inf")
                next_actions.append(torch.argmax(q).item())

            next_actions = torch.tensor(next_actions, device=DEVICE).unsqueeze(1)
            next_q  = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target  = rewards + (1 - dones) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if (self.training_steps > 0 and self.training_steps % self.target_update == 0):
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ------------------------------------------------------------------ #
    #  EVALUATE
    # ------------------------------------------------------------------ #

    def _semi_random_move(self, raw_state):
        """50% minimax / 50% random opponent used during evaluation."""
        valid = valid_moves(raw_state)
        if random.random() < 0.5:
            return minimax_move(raw_state, self.opponent)
        return random.choice(valid)

    def evaluate(self, games: int = 200):
        """Evaluate agent vs 50% minimax / 50% random opponent.
        Returns (win_rate, draw_rate, loss_rate)."""
        wins = draws = losses = 0

        for i in range(games):
            state = self.env.reset()
            done  = False

            if self.fixed_player == 1:
                # agent moves first
                while not done:
                    norm   = normalize_state(state, self.fixed_player)
                    action = self._eval_action(norm)
                    state, _, done = self.env.step(action, player=self.fixed_player)
                    if done:
                        break
                    if not valid_moves(state):
                        break
                    opp_a = self._semi_random_move(state)
                    state, _, done = self.env.step(opp_a, player=self.opponent)

            else:
                # opponent moves first
                while not done:
                    opp_a = self._semi_random_move(state)
                    state, _, done = self.env.step(opp_a, player=self.opponent)
                    if done:
                        break
                    norm   = normalize_state(state, self.fixed_player)
                    action = self._eval_action(norm)
                    state, _, done = self.env.step(action, player=self.fixed_player)

            winner = self.env.check_winner()
            if winner == self.fixed_player:
                wins += 1
            elif winner == self.opponent:
                losses += 1
            else:
                draws += 1

        return wins / games, draws / games, losses / games

    # ------------------------------TRAIN------------------------------------ #

    def train(self):
        tag   = self.phase_label or f"player{'1' if self.fixed_player == 1 else '2'}"
        wins  = 0

        print(f"\n{'='*60}")
        print(f"  Phase: {tag}  |  Role: {'X (first)' if self.fixed_player == 1 else 'O (second)'}")
        print(f"  Episodes: {self.episodes}  |  Device: {DEVICE}")
        print(f"{'='*60}\n")

        for episode in range(self.episodes):
            state = self.env.reset()
            done  = False

            # Agent plays second, so the opponent makes the opening move.
            if self.fixed_player == -1:

                if random.random() < 0.5:
                    opp_a = random.choice(valid_moves(state))
                else:
                    opp_a = self._opponent_action(state)

                state, _, done = self.env.step(opp_a,player=self.opponent)

            while not done:
                # ===== AGENT STEP =====
                norm_before = normalize_state(state, self.fixed_player)
                action      = self._agent_action(norm_before)

                state_after_agent, _, done = self.env.step(action, player=self.fixed_player)
                norm_after = normalize_state(state_after_agent, self.fixed_player)

                reward  = 0.0
                # Reward shaping before terminal rewards are observed.
                reward += 0.05 * self._two_in_row(norm_after)

                if self.fixed_player == 1:
                    reward += self._center_bonus(norm_before, norm_after)

                if done:
                    winner = self.env.check_winner()
                    if winner == self.fixed_player:
                        reward += 1.0
                        wins   += 1
                    elif winner == self.opponent:
                        reward -= 1.0

                    self.buffer.push(norm_before, action, reward, norm_after, True)
                    self.training_steps += 1
                    self._optimize()
                    break

                # ===== OPPONENT STEP =====
                opp_a = self._opponent_action(state_after_agent)
                state_after_opp, _, done = self.env.step(opp_a, player=self.opponent)

                if done:
                    winner = self.env.check_winner()
                    if winner == self.fixed_player:
                        reward += 1.0
                        wins   += 1
                    elif winner == self.opponent:
                        reward -= 1.0

                norm_next = normalize_state(state_after_opp, self.fixed_player)
                self.buffer.push(norm_before, action, reward, norm_next, done)

                state = state_after_opp
                self.training_steps += 1
                self._optimize()

            # ===== END OF EPISODE =====
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if episode % 10 == 0:
                print(f"[{tag}] Episode {episode:>5}  eps={self.epsilon:.3f}")

            # refresh opponent snapshot every 500 episodes
            if episode % 500 == 0 and episode > 0:

                new_state = self.opponent_snapshot.state_dict()

                current = self.policy_net.state_dict()
                # Soft-update the self-play opponent snapshot.
                tau = 0.2

                for k in new_state:
                    new_state[k] = ((1 - tau) * new_state[k] + tau * current[k])

                self.opponent_snapshot.load_state_dict(new_state)

            # Evaluate and save the best-performing checkpoint.
            if episode % 500 == 0 and episode > 0:
                win_r, draw_r, loss_r = self.evaluate(games=500)
                score = win_r + 0.5 * draw_r - loss_r

                print(
                    f"[{tag}] Ep={episode}  "
                    f"Win={win_r:.3f}  Draw={draw_r:.3f}  Lose={loss_r:.3f}  "
                    f"Score={score:.3f}  RecentWins={wins}"
                )

                if score > self.best_score:
                    self.best_score = score
                    torch.save(self.policy_net.state_dict(), f"{self.save_dir}/best_model.pth")
                    print(f"  🔥 New best model saved  (Score={score:.3f})")

                wins = 0

        # save final
        torch.save(self.policy_net.state_dict(), f"{self.save_dir}/final_model.pth")
        print(f"\n[{tag}] Training finished.  Best score: {self.best_score:.3f}")

        return self.policy_net   # return trained model for potential downstream use


if __name__ == "__main__":
    # ---- Phase 1: train as Player 1 (X, moves first) ----
    trainer_p1 = SingleRoleTrainer(
        fixed_player=1,
        episodes=10000,
        phase_label="Phase1_Player1",
    )
    trainer_p1.train()

    # ---- Phase 2: train as Player 2 (O, moves second) — fresh model ----
    trainer_p2 = SingleRoleTrainer(
        fixed_player=-1,
        episodes=10000,
        phase_label="Phase2_Player2",
    )

    trainer_p2.train()

    print("\n✅ Both phases complete.")
    print("   models2/player1/best_model.pth  — plays as X")
    print("   models2/player2/best_model.pth  — plays as O")