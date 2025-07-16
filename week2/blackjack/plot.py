import gym
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

env = gym.make("Blackjack-v1", sab=True)

def evaluate_policy(policy, eval_episodes=10_000):
    wins = draws = losses = 0
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

    win_rate = wins / eval_episodes
    return win_rate


# --- Monte Carlo Training ---
def monte_carlo_train(max_episodes, eval_every=100_000):
    policy = defaultdict(int)
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    gamma = 1.0

    mc_win_rates = []

    for ep in range(1, max_episodes + 1):
        episode = []
        obs, _ = env.reset()
        state = obs
        done = False

        while not done:
            action = random.choice([0, 1])
            next_obs, reward, done, _, _ = env.step(action)
            next_state = next_obs
            episode.append((state, action, reward))
            state = next_state

        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                returns[(s, a)].append(G)
                Q[s][a] = np.mean(returns[(s, a)])
                policy[s] = np.argmax(Q[s])

        if ep % eval_every == 0:
            win_rate = evaluate_policy(policy)
            mc_win_rates.append(win_rate)
            print(f"[MC] Episodes: {ep} | Win Rate: {win_rate:.4f}")

    return mc_win_rates


# --- Q-Learning Training ---
def q_learning_train(max_episodes, eval_every=100_000):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = defaultdict(int)
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1

    q_win_rates = []

    for ep in range(1, max_episodes + 1):
        obs, _ = env.reset()
        state = obs
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_obs, reward, done, _, _ = env.step(action)
            next_state = next_obs

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + gamma * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error

            state = next_state

        if ep % eval_every == 0:
            for s in Q:
                policy[s] = np.argmax(Q[s])
            win_rate = evaluate_policy(policy)
            q_win_rates.append(win_rate)
            print(f"[Q]  Episodes: {ep} | Win Rate: {win_rate:.4f}")

    return q_win_rates


# --- Main Training and Plotting ---
def main():
    max_episodes = 1_000_000
    eval_every = 100_000

    print("Training Monte Carlo...")
    mc_win_rates = monte_carlo_train(max_episodes // 2, eval_every)  # 500k
    print("\nTraining Q-Learning...")
    q_win_rates = q_learning_train(max_episodes, eval_every)  # 1M

    x = list(range(eval_every, max_episodes + 1, eval_every))
    x_mc = list(range(eval_every, max_episodes // 2 + 1, eval_every))

    plt.plot(x_mc, mc_win_rates, label="Monte Carlo", marker='o')
    plt.plot(x, q_win_rates, label="Q-Learning", marker='x')
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.title("Blackjack: Monte Carlo vs Q-Learning")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
