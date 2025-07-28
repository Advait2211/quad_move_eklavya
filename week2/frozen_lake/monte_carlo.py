from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym

import pickle


env = gym.make("FrozenLake-v1")

done = False
# print(env)
observation, info = env.reset() # reset is used to start an episode


"""
observation contains 3 things:
1. The players current sum
2. Value of the dealers face-up card
3. usable ace or not

info is dict with additional information
"""

print(observation, info)


class BlackJackAgent():
    def __init__(self, env,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,):

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_sum = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_count = defaultdict(lambda: np.zeros(env.action_space.n))
        
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.wins = 0
        self.loss = 0
        self.draws = 0
        self.result = []

        self.episode = []


    def get_action(self, env, obs: tuple[int, int, bool]) -> int:


        if np.random.random() < self.epsilon:
            return env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[obs]))
        
        # reduce exploitation as more and more iterations are done. this is defined by the epsilon decay

    def store_transition(self, obs, action, reward):
        self.episode.append((obs, action, reward))


    def update(
        self
    ):  
        G = 0
        visited = set()

        for obs, action, reward in reversed(self.episode):
            G = reward + self.discount_factor * G

            if (obs, action) not in visited:
                visited.add((obs, action))

                self.returns_sum[obs][action] += G
                self.returns_count[obs][action] += 1
                self.q_values[obs][action] = (
                    self.returns_sum[obs][action] / self.returns_count[obs][action]
                )

        # Reset episode memory
        self.episode = []

    def track_rewards(self, reward):
        if reward > 0:
            self.wins += 1
        elif reward < 0:
            self.loss += 1
        else:
            self.draws += 1

        self.result.append(reward)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)




n_episodes = 1_00_000
# n_episodes = 5
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

agent = BlackJackAgent(
    env=env,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)


env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.store_transition(obs, action, reward)

        done = terminated or truncated
        obs = next_obs

    agent.update()
    agent.track_rewards(reward)
    agent.decay_epsilon()

# print(agent.wins, agent.draws, agent.loss)

# wins = agent.wins
# losses = agent.loss
# draws = agent.draws

print(f"Total Games: {n_episodes}")
print(f"Wins: {agent.wins} ({agent.wins / n_episodes * 100:.2f}%)")
print(f"Draws: {agent.draws} ({agent.draws / n_episodes * 100:.2f}%)")
print(f"Losses: {agent.loss} ({agent.loss / n_episodes * 100:.2f}%)")



results = np.array(agent.result)

episodes = np.arange(1, len(results) + 1)
wins = (results == 1).astype(int)
losses = (results == -1).astype(int)

# Cumulative sums
cumulative_wins = np.cumsum(wins)
cumulative_losses = np.cumsum(losses)

plt.figure(figsize=(10, 5))
plt.step(episodes, cumulative_wins, where='post', label="Wins", color="green")
plt.step(episodes, cumulative_losses, where='post', label="Losses", color="red")
plt.xlabel("Episode")
plt.ylabel("Cumulative Count")
plt.title("Cumulative Wins and Losses Over Episodes")
plt.legend()
plt.grid(True)
# plt.show()

print(agent.q_values)


# with open("mc_q_values.pkl", "wb") as f:
#     pickle.dump(agent.q_values, f)



agent.epsilon = 0.0 # 0 exploration, see what the trained model does

eval_episodes = 10_000
wins, losses, draws = 0, 0, 0

for _ in range(eval_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if reward > 0:
        wins += 1
    elif reward < 0:
        losses += 1
    else:
        draws += 1

print(f"Evaluation over {eval_episodes} episodes:")
print(f"Wins: {wins} ({wins / eval_episodes * 100:.2f}%)")
print(f"Losses: {losses} ({losses / eval_episodes * 100:.2f}%)")
print(f"Draws: {draws} ({draws / eval_episodes * 100:.2f}%)")

"""
top score so far:
Evaluation over 10000 episodes:
Wins: 4364 (43.64%)
Losses: 4809 (48.09%)
Draws: 827 (8.27%)

n_episodes = 10_000_000
"""