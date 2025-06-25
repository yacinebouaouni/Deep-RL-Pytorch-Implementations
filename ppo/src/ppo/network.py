import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCriticModel(nn.Module):
    """Actor-Critic Network for PPO with continuous action space."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 64, device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        # Actor network: output activation is tanh
        # Actor network: output activation is tanh
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        ).to(device)
        # Critic network: output activation is None
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ).to(device)

        # Learnable log standard deviation for actions (std ~ 0.6)
        self.actor_log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the value function for given observations using the critic network."""
        return self.critic(obs).squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> tuple[np.ndarray, float]:
        """Select an action in [-1, 1] using a Tanh-transformed Normal distribution (supports multi-dimensional actions)."""
        mean = self.actor(obs)
        std = torch.exp(self.actor_log_std.expand_as(mean)).to(self.device)
        dist = Normal(mean, std)

        action = dist.rsample()  # rsample for reparameterization
        log_prob = dist.log_prob(action).sum(-1)

        action_np = action.detach().cpu().numpy()
        return action_np, log_prob.detach().item()

    def get_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the log probability of a given action under the current policy (multi-dimensional support)."""
        obs = obs.to(self.device)
        action = action.to(self.device)
        mean = self.actor(obs)
        std = torch.exp(self.actor_log_std).to(self.device)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy
