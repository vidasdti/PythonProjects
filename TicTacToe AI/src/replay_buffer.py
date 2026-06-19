from collections import deque
import random
import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer used for DQN training.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""

        self.buffer.append((
            np.array(state, dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        """Sample a random mini-batch of transitions."""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        """Return the current number of stored transitions."""
        return len(self.buffer)