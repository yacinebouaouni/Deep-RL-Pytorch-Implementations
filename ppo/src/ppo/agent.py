import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ppo.config import PPOHyperparameters
from ppo.network import ActorCriticDiscreteModel


class PPOAgent:
    """Implementation of Proximal Policy Optimization (PPO) algorithm."""

    def __init__(
        self,
        env,
        log_dir: str = "ppo_logs",
        hyperparams: PPOHyperparameters = PPOHyperparameters(),
    ):
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        # Use .n for discrete action space
        self.action_dim = self.env.action_space.n
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
        self.coeff_gae_lambda = self.hyperparams.coeff_gae_lambda
        self.minibatch_size = getattr(
            self.hyperparams, "minibatch_size", 64
        )  # Default to 64 if not set

        self.initial_lr = self.lr  # For learning rate annealing
        self.initial_clip_ratio = self.clip_ratio  # Store initial value for annealing
        
        # step 1: Initialize actor (policy) and critic (value) networks
        self.actor_critic = ActorCriticDiscreteModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=128,
            device=self.device,
        )

        # define optimizer for both actor and critic networks (include log_std)
        self.optimizer = torch.optim.Adam(
            list(self.actor_critic.parameters()),
            lr=self.lr,
        )

        # Initialize TensorBoard writer for logging
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    @staticmethod
    def _compute_returns(rewards, terminated, last_value, discount_factor):
        """
        Compute rewards-to-go (returns) for a single episode.
        Args:
            rewards (list or np.ndarray): Rewards for the episode.
            terminated (bool): Whether the episode ended with a terminal state.
            last_value (float): Value estimate for the last state (0 if terminated).
            discount_factor (float): Discount factor (gamma).
        Returns:
            np.ndarray: Returns (rewards-to-go) for each timestep in the episode.
        """
        rewards = np.array(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        dones = np.zeros_like(rewards, dtype=np.float32)
        if terminated:
            dones[-1] = 1.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                returns[t] = rewards[t] + discount_factor * (1 - dones[t]) * last_value
            else:
                returns[t] = (
                    rewards[t] + discount_factor * (1 - dones[t]) * returns[t + 1]
                )
        return returns

    @staticmethod
    def _compute_advantages(returns, values, normalize=True):
        """
        Compute advantages as the difference between returns and value estimates.
        Args:
            returns (torch.Tensor): Returns (rewards-to-go) for each timestep.
            values (torch.Tensor): Value estimates for each timestep.
            normalize (bool): Whether to normalize the advantages.
        Returns:
            torch.Tensor: Advantage estimates for each timestep.
        """
        advantages = returns - values
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return advantages

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
        batch_lengths = []  # Length of each episode
        batch_values_all = []  # Value estimates for each state
        batch_terminated = []  # Terminated flags for each episode
        batch_last_obs = []  # Last obs after episode ends (for bootstrapping)
        batch_last_value = []  # Store value for last obs if not terminated
        batch_returns = []

        timestep = 0  # Current timestep in the batch
        while timestep < self.timesteps_per_batch:
            obs, _ = self.env.reset()
            terminated, truncated = False, False
            episode_rewards = []  # Rewards collected in the current episode
            episode_obs = []  # Observations collected in the current episode
            episode_values = []  # Values estimates for the current episode
            episode_length = 0  # Length of the current episode
            last_value = None
            while (
                not terminated
                and not truncated
                and episode_length < self.max_timesteps_per_episode
                and timestep < self.timesteps_per_batch
            ):
                # select action using the actor network (no grad needed)
                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        obs, dtype=torch.float32, device=self.device
                    )
                    action, log_prob = self.actor_critic.get_action(obs_tensor)
                    value = self.actor_critic.get_value(obs_tensor).detach()

                episode_obs.append(obs)
                batch_actions.append(action.clone())
                batch_log_probs.append(log_prob.clone())
                episode_values.append(value.item())

                # Convert action to Python int for discrete envs
                action_np = int(action.item())
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                episode_rewards.append(reward)

                timestep += 1
                episode_length += 1
                obs = next_obs  # update obs for next step

            episode_length = len(episode_obs)
            batch_lengths.append(episode_length)
            batch_obs.append(episode_obs)
            batch_rewards.append(episode_rewards)
            batch_values_all += episode_values
            batch_terminated.append(terminated)
            batch_last_obs.append(obs)  # store last obs after episode ends
            # Only get value for last obs if not terminated
            if not terminated:
                with torch.no_grad():
                    last_value = self.actor_critic.get_value(
                        torch.tensor(obs, dtype=torch.float32, device=self.device)
                    ).item()
            else:
                last_value = 0.0
            batch_last_value.append(last_value)

        # step 4: compute rewards-to-go (returns)
        batch_returns_list = [
            PPOAgent._compute_returns(r, t, lv, self.discount_factor)
            for r, t, lv in zip(batch_rewards, batch_terminated, batch_last_value)
        ]
        # Extract the last return from each episode (total discounted return per episode)
        episode_final_returns = np.array(
            [ep_returns[-1] for ep_returns in batch_returns_list], dtype=np.float32
        )
        episode_final_returns_mean = episode_final_returns.mean()
        episode_final_returns_std = episode_final_returns.std()
        batch_returns = np.concatenate(batch_returns_list, axis=0)
        batch_returns = torch.as_tensor(
            batch_returns, dtype=torch.float32, device=self.device
        )
        batch_values_all = torch.as_tensor(
            batch_values_all, dtype=torch.float32, device=self.device
        )
        batch_advantages = PPOAgent._compute_advantages(
            batch_returns, batch_values_all, normalize=True
        )

        # Now stack/concatenate for the return values
        batch_obs_flat = torch.as_tensor(
            np.concatenate(batch_obs, axis=0), dtype=torch.float32, device=self.device
        )
        batch_actions = torch.stack(batch_actions).to(torch.long).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(torch.float32).to(self.device)
        return (
            batch_obs_flat,
            batch_actions,
            batch_log_probs,
            batch_advantages,
            batch_returns,
            batch_lengths,
            episode_final_returns_mean,
            episode_final_returns_std,
        )

    def learn(self, total_timesteps: int):
        current_timestep = 0
        with trange(total_timesteps, desc="PPO Training", unit="steps") as pbar:
            while current_timestep < total_timesteps:
                # step 3: collect a batch of experiences
                (
                    batch_obs,
                    batch_actions,
                    batch_log_probs_k,
                    batch_advantages_old,
                    batch_returns,
                    batch_lengths,
                    episode_final_returns_mean,
                    episode_final_returns_std,
                ) = self.rollout()

                batch_size = batch_obs.shape[0]
                minibatch_size = self.minibatch_size
                indices = np.arange(batch_size)

                for epoch in range(self.n_epochs):
                    np.random.shuffle(indices)
                    for start in range(0, batch_size, minibatch_size):
                        end = start + minibatch_size
                        mb_idx = indices[start:end]
                        mb_obs = batch_obs[mb_idx]
                        mb_actions = batch_actions[mb_idx]
                        mb_log_probs_k = batch_log_probs_k[mb_idx]
                        mb_advantages_old = batch_advantages_old[mb_idx]
                        mb_returns = batch_returns[mb_idx]

                        mb_log_probs, entropy = self.actor_critic.get_log_prob_entropy(
                            mb_obs, mb_actions
                        )
                        ratios = torch.exp(mb_log_probs - mb_log_probs_k)
                        surrogate_loss_term1 = ratios * mb_advantages_old
                        surrogate_loss_term2 = (
                            torch.clamp(
                                ratios, 1 - self.clip_ratio, 1 + self.clip_ratio
                            )
                            * mb_advantages_old
                        )
                        actor_loss = -torch.min(
                            surrogate_loss_term1, surrogate_loss_term2
                        ).mean()
                        mb_values = self.actor_critic.get_value(mb_obs)
                        critic_loss = F.mse_loss(mb_values, mb_returns)
                        total_loss = (
                            actor_loss
                            + self.coeff_loss_vf * critic_loss
                            - self.coeff_loss_entropy * entropy.mean()
                        )
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actor_critic.parameters(), max_norm=0.5
                        )
                        self.optimizer.step()

                current_timestep += sum(batch_lengths)
                pbar.update(sum(batch_lengths))

                # Anneal learning rate and clip ratio
                self._update_lr(current_timestep, total_timesteps)
                self._update_clip_ratio(current_timestep, total_timesteps)

                # Log the training progress
                self._log_tensorboard(
                    actor_loss,
                    critic_loss,
                    episode_final_returns_mean,
                    episode_final_returns_std,
                    current_timestep,
                )

                print(
                    f"Step: {current_timestep}, "
                    f"Actor Loss: {actor_loss.item():.4f}, "
                    f"Critic Loss: {critic_loss.item():.4f}, "
                    f"Episode Return Mean: {episode_final_returns_mean:.4f}, "
                    f"Episode Return Std: {episode_final_returns_std:.4f}"
                )
        self.writer.close()

    def _update_lr(self, current_timestep, total_timesteps):
        """Linearly anneal the learning rate."""
        frac = 1.0 - (current_timestep / float(total_timesteps))
        lr = self.initial_lr * frac
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _update_clip_ratio(self, current_timestep, total_timesteps):
        """Linearly anneal the clip ratio from initial value to 0.1."""
        frac = 1.0 - (current_timestep / float(total_timesteps))
        self.clip_ratio = 0.1 + (self.initial_clip_ratio - 0.1) * frac

    def _log_tensorboard(self, actor_loss, critic_loss, episode_final_returns_mean, episode_final_returns_std, current_timestep):
        """
        Log training metrics to TensorBoard.
        Args:
            actor_loss (torch.Tensor or float): Actor loss value.
            critic_loss (torch.Tensor or float): Critic loss value.
            episode_final_returns_mean (float): Mean of episode returns.
            episode_final_returns_std (float): Std of episode returns.
            current_timestep (int): Current timestep in training.
        """
        self.writer.add_scalar("Loss/Actor", actor_loss.item() if hasattr(actor_loss, 'item') else actor_loss, current_timestep)
        self.writer.add_scalar("Loss/Critic", critic_loss.item() if hasattr(critic_loss, 'item') else critic_loss, current_timestep)
        self.writer.add_scalar("EpisodeReturn/Mean", episode_final_returns_mean, current_timestep)
        self.writer.add_scalar("EpisodeReturn/Std", episode_final_returns_std, current_timestep)