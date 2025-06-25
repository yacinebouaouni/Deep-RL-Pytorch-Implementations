from dataclasses import dataclass


@dataclass
class PPOHyperparameters:
    timesteps_per_batch: int = 4096  # Batch size for each update
    max_timesteps_per_episode: int = 256  # Maximum timesteps per episode
    discount_factor: float = 0.99  # Discount factor for rewards
    n_epochs: int = 10  # Number of epochs for each update
    clip_ratio: float = 0.2  # Clipping ratio for PPO
    lr: float = 5e-4  # Learning rate for the optimizer
    coeff_loss_vf: float = 0.5  # Coefficient for value function loss
    coeff_loss_entropy: float = 0.01  # Coefficient for entropy loss
