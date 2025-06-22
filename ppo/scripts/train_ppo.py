import argparse
import gymnasium
import torch
from ppo.agent import PPOAgent


def train_agent(env_name="MountainCarContinuous-v0", total_timesteps=200000):
    env = gymnasium.make(env_name)
    agent = PPOAgent(env)
    agent.learn(total_timesteps=total_timesteps)
    return agent


def record_video(
    agent, env_name="MountainCarContinuous-v0", video_folder="ppo_videos", seed=42
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
            action, _ = agent._select_action(obs_tensor)
        obs, reward, done, truncated, _ = env.step(action)

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
    args = parser.parse_args()

    agent = train_agent(env_name=args.env_name, total_timesteps=args.total_timesteps)
    record_video(agent, env_name=args.env_name)


if __name__ == "__main__":
    main()
