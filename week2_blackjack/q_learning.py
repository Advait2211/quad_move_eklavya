import gym
import numpy as np
from collections import defaultdict
import random

# Create Blackjack environment
env = gym.make("Blackjack-v1", sab=True)

# Q-Table and policy initialization
Q = defaultdict(lambda: np.zeros(env.action_space.n))  # Q[state][action]
policy = defaultdict(int)  # best action for each state

# Hyperparameters
episodes = 1000_000
alpha = 0.1        # learning rate
gamma = 1.0        # discount factor
epsilon = 0.1      # exploration rate

for ep in range(episodes):
    obs, _ = env.reset()
    state = obs  # (player_sum, dealer_card, usable_ace)
    done = False

    while not done:
        # ε-greedy action selection
        if random.random() < epsilon:
            action = env.action_space.sample()  # random action
        else:
            action = np.argmax(Q[state])  # best known action

        next_obs, reward, done, _, _ = env.step(action)
        next_state = next_obs

        # TD target using Q-learning (off-policy)
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += alpha * td_error

        state = next_state

# Derive final policy from Q-values
for state in Q:
    policy[state] = np.argmax(Q[state])

# Evaluation
wins, draws, losses = 0, 0, 0
eval_episodes = 10_000

for _ in range(eval_episodes):
    obs, _ = env.reset()
    state = obs
    done = False

    while not done:
        action = policy.get(state, 1)  # default to hit
        next_obs, reward, done, _, _ = env.step(action)
        state = next_obs

    if reward > 0:
        wins += 1
    elif reward == 0:
        draws += 1
    else:
        losses += 1

# Results
total = wins + draws + losses
print(f"\nEvaluation over {eval_episodes} episodes:")
print(f"Wins:   {wins} ({wins / total:.2%})")
print(f"Draws:  {draws} ({draws / total:.2%})")
print(f"Losses: {losses} ({losses / total:.2%})")
print(f"Win Ratio (wins / total): {wins / total:.4f}")
