from __future__ import annotations
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import random
import torch
import torch.optim as optim
import gymnasium as gym

env = gym.make("LunarLander-v3")
state_size = env.observation_space.shape[0]  # 8
action_size = env.action_space.n             # 4

# replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size=100_000):  # **increased buffer size for stability**
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

# the neural network that will generate q values
class DQN(torch.nn.Module):
    # state space = 8 = input, action space = 4 = output
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# hyperparameters
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 128
BUFFER_SIZE = 1_00000
REPLAY_START_SIZE = 5_000  # **start training only after buffer fills to this size**
eps = 1.0
eps_min = 0.01
eps_decay = 0.995
TARGET_UPDATE_FREQ = 10
NUM_EPISODES = 150
win_count = 0
win_rates = []

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

episode_rewards = []

for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset(seed=episode)[0]
    done = False
    total_reward = 0  # reset every time we start a new episode

    while not done:
        # epsilon greedy
        if random.random() < eps:
            action = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) >= REPLAY_START_SIZE:  # **don't train until buffer is sufficiently full**
            for _ in range(5):  # **train multiple times per step for faster convergence**
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                # Get max Q values for next states from target network (no gradient needed)
                with torch.no_grad():
                    max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]

                # Compute target Q values
                y = rewards + GAMMA * max_next_q * (1 - dones)

                # Get current Q-values for the actions taken
                current_q = policy_net(states).gather(1, actions)

                # Compute the loss between predicted Q and target Q
                loss = torch.nn.functional.mse_loss(current_q, y)

                # Update the policy network
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)  # **clip gradients**
                optimizer.step()

    episode_rewards.append(total_reward)
    if total_reward >= 200:
        win_count += 1

    win_rate = win_count / (episode + 1)
    win_rates.append(win_rate)

    # Decay epsilon after each episode
    eps = max(eps_min, eps * eps_decay)

    # Update target network every few episodes
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())

# --- Plot learning curve ---
# plt.plot(episode_rewards)
# plt.xlabel("Episode")
# plt.ylabel("Total Reward")
# plt.title("DQN Training on LunarLander-v2")
# plt.grid()
# plt.show()

# plt.plot(win_rates)
# plt.xlabel("Episode")
# plt.ylabel("Win Rate")
# plt.title("Win Rate Over Episodes")
# plt.grid(True)
# plt.show()

# --- Evaluation Phase ---
def evaluate_agent(env, policy_net, episodes=20, max_steps=1000):
    wins = 0
    rewards = []
    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0
        while not done and step_count < max_steps:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = policy_net(state_tensor).argmax().item()
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)
        if total_reward >= 200:
            wins += 1
    print(f"\nEvaluation win rate: {wins}/{episodes} = {wins/episodes:.2f}")
    plt.plot(rewards)
    plt.title("Evaluation Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()

evaluate_agent(env, policy_net)  # **evaluate policy after training**
