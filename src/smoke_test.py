"""One minibatch epoch per agent; does not run full RL training."""

import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import gymnasium as gym
from agents import DQN, DoubleDQN, GeneralDoubleDQN
from experiments import smoke_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lunar", action="store_true", help="Also test LunarLander (requires Box2D)."
    )
    args = parser.parse_args()
    cases = [(DQN, "CartPole-v1", 64), (DoubleDQN, "CartPole-v1", 64)]
    if args.lunar:
        cases.append((GeneralDoubleDQN, "LunarLander-v3", 128))
    for cls, env_id, batch in cases:
        env = gym.make(env_id)
        options = dict(
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.01,
            epsilon_max=1.0,
            epsilon_interval=0.99,
            batch_size=batch,
            max_memory_length=50_000,
            pure_exploration_episodes=20 if batch == 128 else 30,
            epsilon_greedy_episodes=500,
        )
        (
            options.update(env=env)
            if cls is GeneralDoubleDQN
            else options.update(num_actions=2)
        )
        try:
            smoke_check(cls(**options), env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
