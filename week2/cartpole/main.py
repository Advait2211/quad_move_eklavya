from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import gymnasium as gym

env = gym.make("CartPole-v1")

done = False

observation, info = env.reset()

print(observation)

class CartPoleAgent():
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor = 1):

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs):

        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    @staticmethod
    def discretize_state(obs, bins=(15, 15, 15, 15)):
        cart_x_bin = np.linspace(-4.8, 4.8, bins[0])
        cart_vel = np.linspace(-5, 5, bins[1])
        pole_angle = np.linspace(-0.418, 0.418, bins[2])
        pole_angle_vel = np.linspace(-5, 5, bins[3])
        cart_x, cart_velo, pole_angle_, pole_angle_velo = obs

        cart_x_bin_idx = np.digitize(cart_x, cart_x_bin)
        cart_vel_idx = np.digitize(cart_velo, cart_vel)
        pole_angle_idx = np.digitize(pole_angle_, pole_angle)
        pole_angle_vel_idx = np.digitize(pole_angle_velo, pole_angle_vel)

        return (cart_x_bin_idx, cart_vel_idx, pole_angle_idx, pole_angle_vel_idx)

        

    def update(self, obs, action, reward, terminated, next_obs):

        next_q_value = (not terminated) * np.max(self.q_values[next_obs])

        temporal_diff = (reward + self.discount_factor * next_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_diff)

        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.1
n_episodes = 10_000
start_epsilon = 1
epsilon_decay = start_epsilon / ((n_episodes)/2)
final_epsilon = 0.1

agent = CartPoleAgent(
    env, learning_rate, start_epsilon, epsilon_decay, final_epsilon
)


env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

successes = 0

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    discrete_obs = agent.discretize_state(obs)

    done = False

    while not done:
        action = agent.get_action(env, discrete_obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        discrete_next_obs = agent.discretize_state(next_obs)

        agent.update(discrete_obs, action, reward, terminated, discrete_next_obs)

        discrete_obs = discrete_next_obs
        done = terminated or truncated

    if info["episode"]["l"] >= 200:
        successes += 1

    agent.decay_epsilon()

print(f"Successes: {successes} / {n_episodes} episodes")

rewards = [r for r in env.return_queue]

plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Reward Over Episodes")
plt.show()


# Evaluation Phase (No exploration, No updates)
test_episodes = 1000
test_successes = 0

agent.epsilon = 0  # Disable exploration during test

for episode in range(test_episodes):
    obs, info = env.reset()
    discrete_obs = agent.discretize_state(obs)
    done = False

    while not done:
        action = agent.get_action(env, discrete_obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        discrete_obs = agent.discretize_state(next_obs)
        done = terminated or truncated

    if terminated:
        test_successes += 1

print(f"Test Success Rate: {test_successes} / {test_episodes} = {test_successes / test_episodes * 100:.2f}%")
