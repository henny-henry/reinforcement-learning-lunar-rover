"""DQN agents extracted from the original project, with shared replay mechanics."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQN:
    """Original three-layer CartPole DQN; historical target initialization retained."""

    def __init__(
        self,
        gamma,
        epsilon_start,
        epsilon_min,
        epsilon_max,
        epsilon_interval,
        batch_size,
        max_memory_length,
        pure_exploration_episodes,
        epsilon_greedy_episodes,
        num_actions,
    ):
        self.prng = np.random.RandomState()
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_interval = epsilon_interval
        self.pure_exploration_episodes = pure_exploration_episodes
        self.epsilon_greedy_episodes = epsilon_greedy_episodes
        self.num_actions = int(num_actions)
        self.batch_size = batch_size
        self.max_memory_length = max_memory_length
        # Define models
        self.model = self.create_q_model()
        self.model_target = self.create_q_model()
        # Historical choice: target starts independently initialized. See research notes.
        # Experience replay buffers
        self.episode_count = 0
        self.action_history = []
        self.state_history = []
        self.state_next_history = []
        self.rewards_history = []
        self.done_history = []
        self.episode_reward_history = []
        self.running_reward_history = []
        # Counters

    def create_q_model(self):
        inputs = layers.Input(shape=(4,))

        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)

        x = layers.Dense(256, activation="relu")(x)

        actions = layers.Dense(self.num_actions, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=actions)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0),
            loss="mse",
        )

        return model

    def act(self, state):
        if self.prng.random() < self.epsilon:
            # Take random action
            return self.prng.choice(self.num_actions)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_vals = self.model(state_tensor, training=False)
            # Take best action
            return tf.argmax(action_vals[0]).numpy()

    def buffer(self, state, action, state_next, reward, done):
        # Save actions and states in replay buffer
        self.action_history.append(action)
        self.state_history.append(state)
        self.state_next_history.append(state_next)
        self.done_history.append(done)
        self.rewards_history.append(reward)

        if len(self.rewards_history) > self.max_memory_length:
            del self.rewards_history[:1]
            del self.state_history[:1]
            del self.state_next_history[:1]
            del self.action_history[:1]
            del self.done_history[:1]

    def update(self):
        if len(self.done_history) < self.batch_size:
            return None
        if len(self.done_history) > self.batch_size:
            # Get indices of samples for replay buffers
            indices = self.prng.choice(
                (len(self.done_history)), size=self.batch_size, replace=False
            )

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.done_history[i]) for i in indices]
            )

            future_rewards = self.model_target.predict(state_next_sample, verbose=0)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + self.gamma * tf.reduce_max(
                future_rewards, axis=1
            ) * (1 - done_sample)

            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, self.num_actions)

            new_q_values = tf.multiply(q_values, (1 - masks)) + tf.multiply(
                tf.expand_dims(updated_q_values, -1), masks
            )

            history = self.model.fit(
                state_sample,
                new_q_values,
                batch_size=self.batch_size,
                shuffle=False,
                verbose=0,
                epochs=1,
            )
            return history.history["loss"][0]

    def decay_parameters(self):
        self.episode_count += 1
        if self.episode_count > self.pure_exploration_episodes:
            self.epsilon = max(
                self.epsilon - self.epsilon_interval / self.epsilon_greedy_episodes,
                self.epsilon_min,
            )


class DoubleDQN(DQN):
    """Select next actions online and evaluate them with the target network."""

    def update(self):
        if len(self.done_history) < self.batch_size:
            return None

        if len(self.done_history) > self.batch_size:
            # Get indices of samples for replay buffers
            indices = self.prng.choice(
                (len(self.done_history)), size=self.batch_size, replace=False
            )

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([self.state_history[i] for i in indices])
            state_next_sample = np.array([self.state_next_history[i] for i in indices])
            rewards_sample = [self.rewards_history[i] for i in indices]
            action_sample = [self.action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(self.done_history[i]) for i in indices]
            )

            q_next_online = self.model(state_next_sample)

            next_actions = tf.argmax(q_next_online, axis=1, output_type=tf.int32)

            q_next_target = self.model_target(state_next_sample)

            selected_qt = tf.gather(q_next_target, next_actions, axis=1, batch_dims=1)

            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + self.gamma * selected_qt * (
                1 - done_sample
            )

            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, self.num_actions)

            new_q_values = tf.multiply(q_values, (1 - masks)) + tf.multiply(
                tf.expand_dims(updated_q_values, -1), masks
            )

            history = self.model.fit(
                state_sample,
                new_q_values,
                batch_size=self.batch_size,
                shuffle=False,
                verbose=0,
                epochs=1,
            )
            return history.history["loss"][0]


class GeneralDoubleDQN(DoubleDQN):
    """Original two-layer DDQN for flat observations and discrete actions."""

    def __init__(
        self,
        gamma,
        epsilon_start,
        epsilon_min,
        epsilon_max,
        epsilon_interval,
        batch_size,
        max_memory_length,
        pure_exploration_episodes,
        epsilon_greedy_episodes,
        env,
    ):
        self.env = env
        self.obs_shape = env.observation_space.shape
        super().__init__(
            gamma,
            epsilon_start,
            epsilon_min,
            epsilon_max,
            epsilon_interval,
            batch_size,
            max_memory_length,
            pure_exploration_episodes,
            epsilon_greedy_episodes,
            num_actions=env.action_space.n,
        )

    def create_q_model(self):
        inputs = layers.Input(shape=self.obs_shape)

        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)

        actions = layers.Dense(self.num_actions, activation="linear")(x)

        model = keras.Model(inputs=inputs, outputs=actions)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=6 * 10 ** (-4), clipnorm=1.0),
            loss=keras.losses.Huber(),
        )

        return model
