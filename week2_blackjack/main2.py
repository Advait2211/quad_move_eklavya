import gym
import numpy as np
from collections import defaultdict
import random

# Create Blackjack environment
env = gym.make("Blackjack-v1", sab=True)

# Initialize Q, policy, and returns
policy = defaultdict(int)
Q = defaultdict(lambda: np.zeros(env.action_space.n))
returns = defaultdict(list)

# Training parameters
episodes = 500_000
gamma = 1.0  # no discounting

for ep in range(episodes):
    episode = []

    obs, _ = env.reset()
    state = obs  # obs is already a tuple (player_sum, dealer_card, usable_ace)

    done = False
    while not done:
        # Exploring start
        action = random.choice([0, 1])  # 0: stick, 1: hit
        next_obs, reward, done, _, _ = env.step(action)
        next_state = next_obs  # also a tuple

        episode.append((state, action, reward))
        state = next_state

    # Monte Carlo update
    G = 0
    visited = set()
    for t in reversed(range(len(episode))):
        state_t, action_t, reward_t = episode[t]
        G = gamma * G + reward_t

        if (state_t, action_t) not in visited:
            visited.add((state_t, action_t))
            returns[(state_t, action_t)].append(G)
            Q[state_t][action_t] = np.mean(returns[(state_t, action_t)])
            policy[state_t] = np.argmax(Q[state_t])

# Evaluation
wins, draws, losses = 0, 0, 0
eval_episodes = 10_000

for _ in range(eval_episodes):
    obs, _ = env.reset()
    state = obs

    done = False
    while not done:
        action = policy.get(state, 1)  # default to hit if unseen
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
