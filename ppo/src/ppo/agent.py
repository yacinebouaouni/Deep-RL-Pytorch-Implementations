import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from ppo.config import PPOHyperparameters
from ppo.network import ActorCriticModel


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
        self.coeff_gae_lambda = self.hyperparams.coeff_gae_lambda

        # step 1: Initialize actor (policy) and critic (value) networks
        self.actor_critic = ActorCriticModel(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dim=128,
            device=self.device,
        )

        # define optimizer for both actor and critic networks (include log_std)
        self.optimizer = torch.optim.Adam(
            list(self.actor_critic.parameters()),
            lr=self.lr,
            eps=1e-5,
        )

        # Initialize TensorBoard writer for logging
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))




    def compute_gae(
        self,
        batch_obs: list[list[float]],
        batch_rewards: list[list[float]],
        batch_terminated: list[bool],
        batch_last_obs: list[np.ndarray],
        gamma: float,
        lambd: float,
    ) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE) for the batch of observations and rewards.
        """
        all_advantages = []

        for obs_seq, reward_seq, terminated, last_obs in zip(batch_obs, batch_rewards, batch_terminated, batch_last_obs):
            obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
            values = self.actor_critic.get_value(obs_tensor.to(self.device)).cpu().detach().numpy()  # [T]
            rewards = np.array(reward_seq, dtype=np.float32)
            T = len(rewards)
            
            # Bootstrap value: 0 if terminated, otherwise value of last_obs
            if terminated:
                next_value = 0.0
            else:
                last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32)
                next_value = self.actor_critic.get_value(last_obs_tensor.to(self.device)).cpu().detach().item()
            
            # Pad the values with the next_value at the end
            values = np.append(values, next_value)  # [T+1]
            
            advantages = np.zeros(T, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * lambd * gae
                advantages[t] = gae
            
            all_advantages.append(advantages)
        
        # Flatten to a 1D tensor for the entire batch
        flat_advantages = np.concatenate(all_advantages)
        
        flat_advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-10)
        return flat_advantages


    # def compute_gae(
    #     self,
    #     batch_obs: list[list[float]],
    #     batch_rewards: list[list[float]],
    #     batch_terminated: list[bool],
    #     batch_last_obs: list[np.ndarray],
    #     gamma: float,
    #     lambd: float,
    # ) -> torch.Tensor:
    #     """Compute Generalized Advantage Estimation (GAE) for the batch of observations and rewards.
        
    #     Args:
    #         batch_obs (list[list[float]]): List of lists containing observations for each episode.
    #         batch_rewards (list[list[float]]): List of lists containing rewards for each episode.
    #         batch_terminated (list[bool]): List of booleans indicating if each episode is done.
    #         batch_last_obs (list[np.ndarray]): List of last observations for each episode (for bootstrapping).
    #         gamma (float): Discount factor for future rewards.
    #         lambd (float): Lambda parameter for GAE.
            
    #     Returns:
    #         torch.Tensor: Tensor containing the computed advantages for the batch of observations.
    #     """
    #     batch_advantages = []
    #     idx = 0
    #     for episode_obs, episode_rewards, terminated, last_obs in zip(
    #         batch_obs, batch_rewards, batch_terminated, batch_last_obs
    #     ):
    #         episode_obs_copy = episode_obs.copy()
    #         if not terminated:
    #             episode_obs_copy.append(last_obs)
    #         obs_tensor = torch.tensor(
    #             episode_obs_copy, dtype=torch.float32, device=self.device
    #         )
    #         with torch.no_grad():
    #             episode_values = self.actor_critic.get_value(obs_tensor).cpu().numpy()

    #         rewards = np.array(episode_rewards, dtype=np.float32)
    #         values = episode_values[:-1] if not terminated else episode_values
    #         vals_last = episode_values[-1] if not terminated else 0.0
    #         dones = np.zeros_like(rewards, dtype=np.float32)
    #         if terminated:
    #             dones[-1] = 1.0

    #         advantages = np.zeros_like(rewards, dtype=np.float32)
    #         for t in reversed(range(len(rewards))):
    #             if t == len(rewards) - 1:
    #                 td_error = rewards[t] + gamma * (1 - dones[t]) * vals_last - values[t]
    #             else:
    #                 td_error = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
    #             advantages[t] = advantages[t] * lambd * gamma * (1 - dones[t]) + td_error
    #         batch_advantages.append(advantages)
    #         idx += len(rewards)

    #     batch_advantages = np.concatenate(batch_advantages, axis=0)
    #     batch_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=self.device)
    #     batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-10)
    #     return batch_advantages

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

        timestep = 0  # Current timestep in the batch
        while timestep < self.timesteps_per_batch:
            obs, _ = self.env.reset()
            terminated = False
            truncated = False
            episode_length = 0
            episode_rewards = []  # Rewards collected in the current episode
            episode_obs = []  # Observations collected in the current episode
            episode_values = []  # Values estimates for the current episode
            while (
                not terminated
                and not truncated
                and episode_length < self.max_timesteps_per_episode
                and timestep < self.timesteps_per_batch
            ):
                # select action using the actor network (no grad needed)
                with torch.no_grad():
                    action, log_prob = self.actor_critic.get_action(
                        torch.tensor(obs, dtype=torch.float32, device=self.device)
                    )
                    value = self.actor_critic.get_value(
                        torch.tensor(obs, dtype=torch.float32, device=self.device)
                    )

                episode_obs.append(obs)
                batch_actions.append(action.detach().clone())
                batch_log_probs.append(log_prob.detach().clone())
                episode_values.append(value.item())

                # Convert action to numpy for env.step
                action_np = action.detach().cpu().numpy()
                next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
                episode_rewards.append(reward)

                episode_length += 1
                timestep += 1
                obs = next_obs  # update obs for next step



            batch_lengths.append(episode_length)
            batch_rewards.append(episode_rewards)
            batch_obs.append(episode_obs)
            batch_values_all += episode_values
            batch_terminated.append(terminated)
            batch_last_obs.append(obs)  # store last obs after episode ends

        # step 4: compute rewards-to-go (returns)
        batch_returns = []
        for episode_rewards, terminated, last_obs, episode_obs in zip(
            batch_rewards, batch_terminated, batch_last_obs, batch_obs
        ):
            rewards = np.array(episode_rewards, dtype=np.float32)
            returns = np.zeros_like(rewards, dtype=np.float32)
            # Bootstrapping value for last state
            obs_tensor = torch.tensor(
                episode_obs + ([last_obs] if not terminated else []), dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                episode_values = self.actor_critic.get_value(obs_tensor).cpu().numpy()
            vals_last = episode_values[-1] if not terminated else 0.0
            dones = np.zeros_like(rewards, dtype=np.float32)
            if terminated:
                dones[-1] = 1.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    returns[t] = rewards[t] + self.discount_factor * (1 - dones[t]) * vals_last
                else:
                    returns[t] = rewards[t] + self.discount_factor * (1 - dones[t]) * returns[t + 1]
            batch_returns.append(returns)
        batch_returns = np.concatenate(batch_returns, axis=0)
        batch_returns = torch.tensor(batch_returns, dtype=torch.float32, device=self.device)

        batch_values_all = torch.tensor(
            batch_values_all, dtype=torch.float32, device=self.device
        )

        # Compute advantages using the original (list of lists) batch_obs, batch_rewards, etc.
        batch_advantages = self.compute_gae(
            batch_obs=batch_obs,  # list of lists
            batch_rewards=batch_rewards,
            batch_terminated=batch_terminated,
            batch_last_obs=batch_last_obs,
            #batch_values_all=batch_values_all,
            gamma=self.discount_factor,
            lambd=self.coeff_gae_lambda,
        )

        # Now stack/concatenate for the return values
        batch_obs_flat = np.concatenate(batch_obs, axis=0)
        batch_obs_flat = torch.tensor(batch_obs_flat, dtype=torch.float32, device=self.device)
        batch_actions = torch.stack(batch_actions).to(torch.float32).to(self.device)
        batch_log_probs = torch.stack(batch_log_probs).to(torch.float32).to(self.device)
        return (
            batch_obs_flat,
            batch_actions,
            batch_log_probs,
            batch_advantages,
            batch_returns,
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
                    batch_advantages_old,
                    batch_returns,
                    batch_lengths,
                ) = self.rollout()

                returns = batch_returns  # Use returns directly from rollout

                # No need to re-normalize batch_advantages_old (already normalized)

                # step 6: update actor and critic networks
                for epoch in range(self.n_epochs):
                    batch_log_probs, entropy = self.actor_critic.get_log_prob_entropy(
                        batch_obs, batch_actions.to(self.device)
                    )
                    ratios = torch.exp(batch_log_probs - batch_log_probs_old)

                    # calculate surrogate loss
                    surrogate_loss_term1 = ratios * batch_advantages_old
                    surrogate_loss_term2 = (
                        torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        * batch_advantages_old
                    )
                    # Actor loss (policy loss)
                    actor_loss = -torch.min(
                        surrogate_loss_term1, surrogate_loss_term2
                    ).mean()

                    # Critic loss (value function loss)
                    batch_values = self.actor_critic.get_value(batch_obs)
                    critic_loss = F.mse_loss(batch_values, returns)

                    # Combine losses (weighted sum)
                    total_loss = (
                        actor_loss
                        + self.coeff_loss_vf * critic_loss
                        #- self.coeff_loss_entropy * entropy.mean()
                    )

                    # Single optimizer step for both actor and critic
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.parameters(), max_norm=.5
                    )
                    self.optimizer.step()

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
                    "Returns/Mean", returns.mean().item(), current_timestep
                )
                self.writer.add_scalar(
                    "Returns/Std", returns.std().item(), current_timestep
                )
                print(
                    f"Step: {current_timestep}, "
                    f"Actor Loss: {actor_loss.item():.4f}, "
                    f"Critic Loss: {critic_loss.item():.4f}, "
                    f"Returns Mean: {returns.mean().item():.4f}, "
                    f"Returns Std: {returns.std().item():.4f}"
                )
        self.writer.close()
