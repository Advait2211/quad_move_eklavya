import torch
import random
import numpy as np
import gymnasium as gym
from collections import deque

# --- DQN definition (same as training) ---
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- Load environment ---
env = gym.make("LunarLander-v3", render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Load trained model ---
policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load("dqn_lunarlander.pth"))
policy_net.eval()

# --- Evaluate ---
NUM_TEST_EPISODES = 200
win_count = 0
rewards_list = []

for episode in range(NUM_TEST_EPISODES):
    state = env.reset(seed=episode)[0]
    done = False
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    rewards_list.append(total_reward)
    if total_reward >= 200:  # Win condition
        win_count += 1

win_rate = win_count / NUM_TEST_EPISODES
print(f"Average Reward: {np.mean(rewards_list):.2f}")
print(f"Win Rate over {NUM_TEST_EPISODES} episodes: {win_rate:.2%}")
