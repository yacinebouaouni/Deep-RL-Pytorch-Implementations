import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ppo.config import PPOHyperparameters
from ppo.network import FeedForwardNetwork


class PPOAgent:
    """Implementation of Proximal Policy Optimization (PPO) algorithm."""

    def __init__(self, env, log_dir: str = "ppo_logs", hyperparams: PPOHyperparameters = PPOHyperparameters()):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hyperparams = hyperparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Assign hyperparameters from dataclass
        self.timesteps_per_batch = self.hyperparams.timesteps_per_batch
        self.max_timesteps_per_episode = self.hyperparams.max_timesteps_per_episode
        self.discount_factor = self.hyperparams.discount_factor
        self.n_epochs = self.hyperparams.n_epochs
        self.clip_ratio = self.hyperparams.clip_ratio
        self.lr = self.hyperparams.lr
        self.coeff_loss_vf = self.hyperparams.coeff_loss_vf
        self.coeff_loss_entropy = self.hyperparams.coeff_loss_entropy

        # Learnable log standard deviation for actions
        self.log_std = torch.nn.Parameter(torch.full((self.action_dim,), -0.5))  # std ≈ 0.6

        # step 1: Initialize actor (policy) and critic (value) networks
        self.actor = FeedForwardNetwork(self.obs_dim, self.action_dim, 64).to(
            self.device
        )
        self.critic = FeedForwardNetwork(self.obs_dim, 1, 64).to(self.device)

        # define optimizer for both actor and critic networks (include log_std)
        self.actor_optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + [self.log_std], lr=self.lr
        )
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # Initialize TensorBoard writer for logging
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    def _select_action(self, obs: torch.Tensor) -> tuple[np.ndarray, float]:
        """Select an action based on the current observation using the actor network.

        The action is sampled from a Gaussian distribution parameterized by the actor network.
        The log probability of the action is also computed for later use in policy updates.
        """
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        cov_mat = torch.diag(std**2).to(self.device)
        distribution = MultivariateNormal(mean, cov_mat)
        action = distribution.sample()  # Sample an action from the distribution
        log_prob = distribution.log_prob(
            action
        )  # Compute the log probability of the action
        return (
            action.detach().cpu().numpy(),
            log_prob.detach().item(),
        )  # Return action as numpy array and log probability

    def _evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluate the value of the current observation using the critic network.

        The critic network outputs a value estimate for the given observation.
        This value will be used to compute advantages and update the policy.
        """
        return self.critic(obs).squeeze()

    def _evaluate_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Evaluate the log probability of a given action under the current policy.

        This is used to compute the ratio of probabilities for the PPO objective.
        The log probability is computed using the actor network and the covariance matrix.
        """
        mean = self.actor(obs)
        std = torch.exp(self.log_std)
        cov_mat = torch.diag(std**2).to(self.device)
        distribution = MultivariateNormal(mean, cov_mat)
        return distribution.log_prob(action)

    def _compute_rewards_to_go(self, batch_rewards: list[list[float]]) -> list[float]:
        """Compute the returns (rewards-to-go) for each episode.

        The rewards-to-go are computed by discounting future rewards.
        This is done by iterating through the rewards in reverse order and applying the discount factor.
        """
        batch_returns = []
        for rewards in batch_rewards:
            returns = []
            cumulative_return = 0.0
            for reward in reversed(rewards):
                cumulative_return = reward + self.discount_factor * cumulative_return
                returns.append(cumulative_return)
            returns.reverse()  # Reverse the list to match the original order
            batch_returns += returns  # Append the returns for this episode to the batch
        return batch_returns  # Returns a list of lists, where each inner list contains the returns

    def rollout(self):
        """Collect a batch of experiences from the environment.

        We run the environment for a fixed number of timesteps, collecting observations,
        actions, log probabilities, rewards, and lengths of episodes.
        The collected data will be used for training the actor and critic networks.
        """
        batch_obs = []  # Observations collected in the batch
        batch_actions = []  # Actions taken in the batch
        batch_log_probs = []  # Log probabilities of actions taken
        batch_rewards = []  # Rewards collected in the batch
        batch_rewards_to_go = []  # Discounted rewards
        batch_lengths = []  # Length of each episode

        timestep = 0

        while timestep < self.timesteps_per_batch:
            obs, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_length = 0
            episode_rewards = []  # Rewards collected in the current episode

            while (
                not terminated
                and not truncated
                and episode_length < self.max_timesteps_per_episode
            ):
                # collect observation
                batch_obs.append(obs)

                # select action using the actor network
                action, log_prob = self._select_action(
                    torch.tensor(obs, dtype=torch.float32, device=self.device)
                )
                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                # take action in the environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_rewards.append(reward)

                episode_length += 1
                timestep += 1

            batch_lengths.append(
                episode_length
            )  # Store the length of the episode (+1 starts from 0)
            batch_rewards.append(episode_rewards)

        # convert collected data to tensors
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float32, device=self.device)
        batch_actions = torch.tensor(
            batch_actions, dtype=torch.float32, device=self.device
        )
        batch_log_probs = torch.tensor(
            batch_log_probs, dtype=torch.float32, device=self.device
        )

        # step 4: compute rewards-to-go
        batch_rewards_to_go = self._compute_rewards_to_go(batch_rewards)
        batch_rewards_to_go = torch.tensor(
            batch_rewards_to_go, dtype=torch.float32, device=self.device
        )

        return (
            batch_obs,
            batch_actions,
            batch_log_probs,
            batch_rewards_to_go,
            batch_lengths,
        )

    def learn(self, total_timesteps: int):
        current_timestep = 0
        with trange(total_timesteps, desc="PPO Training", unit="steps") as pbar:
            while current_timestep < total_timesteps:
                # step 3: collect a batch of experiences
                (
                    batch_obs,
                    batch_actions,
                    batch_log_probs_old,
                    batch_rewards_to_go,
                    batch_lengths,
                ) = self.rollout()

                # calculate Value estimates for the batch observations
                batch_values = self._evaluate(batch_obs.to(self.device))

                # step 5: compute advantages
                batch_advantages_k = batch_rewards_to_go - batch_values.detach()

                # Normalize advantages
                # This helps stabilize training by ensuring that the advantages have a mean of 0 and a
                batch_advantages_k = (
                    batch_advantages_k - batch_advantages_k.mean()
                ) / (batch_advantages_k.std() + 1e-10)

                # step 6: update actor and critic networks
                for epoch in range(self.n_epochs):
                    batch_log_probs = self._evaluate_action(
                        batch_obs, batch_actions.to(self.device)
                    )
                    ratios = torch.exp(batch_log_probs - batch_log_probs_old)

                    # calculate surrogate loss
                    surrogate_loss_term1 = ratios * batch_advantages_k
                    surrogate_loss_term2 = (
                        torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        * batch_advantages_k
                    )
                    # Actor loss (policy loss)
                    actor_loss = -torch.min(
                        surrogate_loss_term1, surrogate_loss_term2
                    ).mean()

                    # Critic loss (value function loss)
                    batch_values = self._evaluate(batch_obs)
                    critic_loss = F.mse_loss(batch_values, batch_rewards_to_go)

                    # Reconstruct distribution to compute entropy
                    mean = self.actor(batch_obs)
                    std = torch.exp(self.log_std)
                    cov_mat = torch.diag(std**2).to(self.device)
                    distribution = MultivariateNormal(mean, cov_mat)
                    entropy = distribution.entropy().mean()
                    
                    # Combine losses (weighted sum)
                    total_loss = actor_loss + self.coeff_loss_vf * critic_loss - self.coeff_loss_entropy * entropy

                    # Single optimizer step for both actor and critic
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                current_timestep += sum(batch_lengths)
                pbar.update(sum(batch_lengths))

                # Log the training progress
                self.writer.add_scalar(
                    "Loss/Actor", actor_loss.item(), current_timestep
                )
                self.writer.add_scalar(
                    "Loss/Critic", critic_loss.item(), current_timestep
                )
                self.writer.add_scalar(
                    "Rewards/Mean", batch_rewards_to_go.mean().item(), current_timestep
                )
                self.writer.add_scalar(
                    "Rewards/Std", batch_rewards_to_go.std().item(), current_timestep
                )
                
        # write final model parameters to TensorBoard
        self.writer.add_hparams(
            {
                "timesteps_per_batch": self.timesteps_per_batch,
                "max_timesteps_per_episode": self.max_timesteps_per_episode,
                "discount_factor": self.discount_factor,
                "n_epochs": self.n_epochs,
                "clip_ratio": self.clip_ratio,
                "lr": self.lr,
                "coeff_loss_vf": self.coeff_loss_vf,
                "coeff_loss_entropy": self.coeff_loss_entropy,
            },
            {
                "Loss/Actor": actor_loss.item(),
                "Loss/Critic": critic_loss.item(),
                "Rewards/Mean": batch_rewards_to_go.mean().item(),
                "Rewards/Std": batch_rewards_to_go.std().item(),
            },
        )
        self.writer.close()
