import gymnasium
from ppo.agent import PPOAgent


def main():
    env = gymnasium.make("MountainCarContinuous-v0")
    agent = PPOAgent(env)

    # Start training the agent
    agent.learn(total_timesteps=200000)


if __name__ == "__main__":
    main()
