import torch
import numpy as np


def compute_gae(
    actor_critic,
    device,
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

    for obs_seq, reward_seq, terminated, last_obs in zip(
        batch_obs, batch_rewards, batch_terminated, batch_last_obs
    ):
        obs_tensor = torch.tensor(obs_seq, dtype=torch.float32)
        values = (
            actor_critic.get_value(obs_tensor.to(device)).cpu().detach().numpy()
        )  # [T]
        rewards = np.array(reward_seq, dtype=np.float32)
        T = len(rewards)

        # Bootstrap value: 0 if terminated, otherwise value of last_obs
        if terminated:
            next_value = 0.0
        else:
            last_obs_tensor = torch.tensor(last_obs, dtype=torch.float32)
            next_value = (
                actor_critic.get_value(last_obs_tensor.to(device)).cpu().detach().item()
            )

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

    flat_advantages = torch.tensor(flat_advantages, dtype=torch.float32, device=device)

    # Normalize advantages
    flat_advantages = (flat_advantages - flat_advantages.mean()) / (
        flat_advantages.std() + 1e-10
    )
    return flat_advantages
