import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple Deep Q-Network for TicTacToe (9-state input, 9-action output).
    """
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.net(x)