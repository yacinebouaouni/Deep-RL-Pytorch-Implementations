import argparse
import gymnasium
import torch
from ppo.agent import PPOAgent
from ppo.config import PPOHyperparameters


def train_agent(
    env_name="MountainCarContinuous-v0", total_timesteps=200000, hyperparams=None
):
    env = gymnasium.make(env_name)
    if hyperparams is None:
        agent = PPOAgent(env)
    else:
        agent = PPOAgent(env, hyperparams=hyperparams)
    agent.learn(total_timesteps=total_timesteps)
    return agent


def record_video(
    agent, env_name="MountainCarContinuous-v0", video_folder="ppo_logs_videos", seed=42
):
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix="ppo_agent",
    )

    obs, _ = env.reset(seed=seed)
    done = False
    truncated = False

    while not done and not truncated:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            action, _ = agent.actor_critic.get_action(obs_tensor)
        obs, reward, done, truncated, _ = env.step(action.cpu().numpy())

    env.close()
    print(f"Video saved in ./{video_folder}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="MountainCarContinuous-v0",
        help="Gym environment name",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=200000,
        help="Total timesteps for training",
    )
    parser.add_argument(
        "--timesteps_per_batch",
        type=int,
        default=4096,
        help="Batch size for each update",
    )
    parser.add_argument(
        "--max_timesteps_per_episode",
        type=int,
        default=256,
        help="Maximum timesteps per episode",
    )
    parser.add_argument(
        "--discount_factor",
        type=float,
        default=0.99,
        help="Discount factor for rewards",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of epochs for each update",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="Clipping ratio for PPO",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--coeff_loss_vf",
        type=float,
        default=0.5,
        help="Coefficient for value function loss",
    )
    parser.add_argument(
        "--coeff_loss_entropy",
        type=float,
        default=0.01,
        help="Coefficient for entropy loss",
    )
    parser.add_argument(
        "--coeff_gae_lambda",
        type=float,
        default=0.99,
        help="GAE lambda for advantage estimation",
    )
    args = parser.parse_args()

    hyperparams = PPOHyperparameters(
        timesteps_per_batch=args.timesteps_per_batch,
        max_timesteps_per_episode=args.max_timesteps_per_episode,
        discount_factor=args.discount_factor,
        n_epochs=args.n_epochs,
        clip_ratio=args.clip_ratio,
        lr=args.lr,
        coeff_loss_vf=args.coeff_loss_vf,
        coeff_loss_entropy=args.coeff_loss_entropy,
        coeff_gae_lambda=args.coeff_gae_lambda,
    )

    agent = train_agent(
        env_name=args.env_name,
        total_timesteps=args.total_timesteps,
        hyperparams=hyperparams,
    )
    record_video(agent, env_name=args.env_name)


if __name__ == "__main__":
    main()
