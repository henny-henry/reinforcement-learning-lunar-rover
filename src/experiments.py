"""Bounded execution checks and an opt-in driver for the original RL experiments."""

import csv
from pathlib import Path
import numpy as np


def smoke_check(agent, env):
    """Collect one replay batch, perform exactly one epoch, and check inference."""
    # The historical update requires strictly more than batch_size transitions.
    state, _ = env.reset(seed=42)
    agent.prng.seed(42)
    for _ in range(agent.batch_size + 1):
        action = int(agent.act(state))
        next_state, reward, terminated, truncated, _ = env.step(action)
        agent.buffer(state, action, next_state, reward, terminated)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
    before = int(agent.model.optimizer.iterations.numpy())
    loss = agent.update()  # One minibatch, epochs=1; no other training calls.
    after = int(agent.model.optimizer.iterations.numpy())
    if loss is None or not np.isfinite(loss) or after - before != 1:
        raise RuntimeError("Expected exactly one optimizer update with finite loss.")
    agent.epsilon = 0.0
    action = int(agent.act(state))
    if not env.action_space.contains(action):
        raise RuntimeError("Agent returned an invalid action.")
    agent.model_target.set_weights(agent.model.get_weights())
    print(
        f"PASS {type(agent).__name__} / {env.spec.id}: "
        f"{agent.batch_size + 1} transitions, one minibatch epoch, loss={loss:.6f}"
    )
    return float(loss)


def run_experiment(
    agent, env, *, max_episodes, update_target_every, solve_threshold, output
):
    """Original replay/update schedule, with an explicit episode cap and fresh CSV."""
    if max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rewards, running_history = [], []
    steps = 0
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode", "episode_reward", "running_reward", "epsilon"])
        for episode in range(1, max_episodes + 1):
            state, _ = env.reset()
            episode_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = int(agent.act(state))
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Preserve the original distinction: only termination masks the target.
                agent.buffer(state, action, next_state, reward, terminated)
                agent.update()
                state = next_state
                episode_reward += reward
                steps += 1
                if steps % update_target_every == 0:
                    agent.model_target.set_weights(agent.model.get_weights())
            agent.decay_parameters()
            rewards.append(episode_reward)
            rewards = rewards[-100:]
            running_reward = float(np.mean(rewards))
            running_history.append(running_reward)
            writer.writerow([episode, episode_reward, running_reward, agent.epsilon])
            handle.flush()
            print(
                f"Running reward: {running_reward:.2f} at episode {episode}, eps={agent.epsilon:.4f}"
            )
            if running_reward > solve_threshold:
                print(f"Project stopping threshold reached at episode {episode}.")
                break
    return running_history
