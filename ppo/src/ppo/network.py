import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution, TanhTransform


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticModel(nn.Module):
    """Actor-Critic Network for PPO with continuous action space."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 64, device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        # Actor network: output activation is tanh
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        ).to(device)
        # Critic network: output activation is None
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        ).to(device)

        # Learnable log standard deviation for actions (std = 1.0)
        self.actor_log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the value function for given observations using the critic network."""
        return self.critic(obs).squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.actor_log_std.expand_as(mean)).to(self.device)
        base_dist = Normal(mean, std)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def get_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.actor(obs)
        std = torch.exp(self.actor_log_std.expand_as(mean)).to(self.device)
        base_dist = Normal(mean, std)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.base_dist.entropy().sum(-1)
        return log_prob, entropy


class Network(nn.Module):
    def __init__(self, obs_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.PReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class ActorCriticDiscreteModel(nn.Module):
    """Actor-Critic Network for PPO with discrete action space (e.g., LunarLander-v3)."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 128, device: str = "cpu"
    ):
        super().__init__()
        self.device = device
        # Actor network: outputs logits for discrete actions
        self.actor = Network(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        ).to(device)
        # Critic network: output activation is None
        self.critic = Network(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=1,  # Single value output for critic
        ).to(device)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the value function for given observations using the critic network."""
        return self.critic(obs).squeeze(-1)

    def get_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        action_prob = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=action_prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # Ensure action is always a scalar tensor for discrete envs
        return action.detach(), log_prob.detach()

    def get_log_prob_entropy(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        dist = torch.distributions.Categorical(logits=logits)
        # Ensure action is long type for Categorical
        action = action.long()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy
