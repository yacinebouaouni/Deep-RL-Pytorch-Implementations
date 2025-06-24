import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
