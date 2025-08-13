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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)
    

#the neural network that will generate q values
class DQN(torch.nn.Module):
    #state space = 8 = input, action space = 4 = output
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64
BUFFER_SIZE = 50_000
WARMUP_STEPS = 1000
eps = 1.0
eps_min = 0.01
eps_decay = 0.999
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 2500
win_count = 0
win_rates = []

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

episode_rewards = []
update_steps = 0  # step counter outside episodes

for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset(seed=episode)[0]
    done = False
    total_reward = 0

    while not done:
        # epsilon greedy action
        if random.random() < eps:
            action = random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)
                action = q_values.argmax().item()

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) >= WARMUP_STEPS:
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            states      = torch.FloatTensor(states).to(device)
            actions     = torch.LongTensor(actions).unsqueeze(1).to(device)
            rewards     = torch.FloatTensor(rewards).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(next_states).to(device)
            dones       = torch.FloatTensor(dones).unsqueeze(1).to(device)

            with torch.no_grad():
                max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
            y = rewards + GAMMA * max_next_q * (1 - dones)
            current_q = policy_net(states).gather(1, actions)

            loss = torch.nn.functional.mse_loss(current_q, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Increment step counter and update target net if needed
        update_steps += 1
        if update_steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    episode_rewards.append(total_reward)
    if total_reward >= 200:
        win_count += 1

    win_rate = win_count / (episode + 1)
    win_rates.append(win_rate)
    eps = max(eps_min, eps * eps_decay)

# --- Plot learning curve ---
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training on LunarLander-v2")
plt.grid()
plt.show()

print(win_rate)
plt.plot(win_rates)
plt.xlabel("Episode")
plt.ylabel("Win Rate")
plt.title("Win Rate Over Episodes")
plt.grid(True)
plt.show()

# Save trained model
torch.save(policy_net.state_dict(), "dqn_lunarlander.pth")
print("Model saved as dqn_lunarlander.pth")