import gymnasium as gym
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

done = False

observation, info = env.reset()

print(observation)

class MoutainCarAgent():
    def __init__(self, env, learning_rate, initial_epsilon, epsilon_decay, final_epsilon, discount_factor = 0.95):

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
    def discretize_state(obs, bins=(20, 20)):
        pos_bins = np.linspace(-1.2, 0.6, bins[0])
        vel_bins = np.linspace(-0.07, 0.07, bins[1])

        pos, vel = obs
        pos_idx = np.digitize(pos, pos_bins)
        vel_idx = np.digitize(vel, vel_bins)

        return (pos_idx, vel_idx)

        

    def update(self, obs, action, reward, terminated, next_obs):

        next_q_value = (not terminated) * np.max(self.q_values[next_obs])

        temporal_diff = (reward + self.discount_factor * next_q_value - self.q_values[obs][action])

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_diff)

        self.training_error.append(temporal_diff)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.01
n_episodes = 10_000
start_epsilon = 1
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = 0.1

agent = MoutainCarAgent(
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

    if terminated:   # Reached goal
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
test_episodes = 100
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
