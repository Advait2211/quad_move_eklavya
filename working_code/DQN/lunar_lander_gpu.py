from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from tqdm import tqdm
import random
import torch
import torch.optim as optim
import gymnasium as gym

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Parallel Environment Creation ---
NUM_ENVS = 8  # number of environments to run in parallel

def make_env():
    def _thunk():
        return gym.make("LunarLander-v3")
    return _thunk

env = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
state_size = env.single_observation_space.shape[0]
action_size = env.single_action_space.n

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, buffer_size=100000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones

    def __len__(self):
        return len(self.buffer)

# --- DQN Network ---
class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super().__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- Hyperparameters ---
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 512
BUFFER_SIZE = 200_000
WARMUP_STEPS = 5000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
TARGET_UPDATE_FREQ = 2000
TOTAL_STEPS = 300_000  # total training timesteps

# --- Initialize ---
policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

episode_rewards = np.zeros(NUM_ENVS)
total_rewards_history = []
epsilon = EPSILON_START
update_steps = 0

# --- Training Loop ---
state, _ = env.reset(seed=None)

for step in tqdm(range(TOTAL_STEPS)):
    # Epsilon-greedy
    if random.random() < epsilon:
        actions = np.array([random.randint(0, action_size - 1) for _ in range(NUM_ENVS)])
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            q_values = policy_net(state_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()

    next_state, reward, terminated, truncated, _ = env.step(actions)
    done = np.logical_or(terminated, truncated)

    # Store experience
    replay_buffer.add(state, actions, reward, next_state, done)
    state = next_state

    # Track rewards for plotting
    episode_rewards += reward
    for i in range(NUM_ENVS):
        if done[i]:
            total_rewards_history.append(episode_rewards[i])
            episode_rewards[i] = 0.0

    # Train
    if len(replay_buffer) >= WARMUP_STEPS:
        states, actions_b, rewards_b, next_states, dones_b = replay_buffer.sample(BATCH_SIZE)

        states      = torch.FloatTensor(states).to(device)
        actions_b   = torch.LongTensor(actions_b).unsqueeze(1).to(device)
        rewards_b   = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones_b     = torch.FloatTensor(dones_b).unsqueeze(1).to(device)

        with torch.no_grad():
            max_next_q = target_net(next_states).max(dim=1, keepdim=True)[0]
            y = rewards_b + GAMMA * max_next_q * (1 - dones_b)

        current_q = policy_net(states).gather(1, actions_b)

        loss = torch.nn.functional.mse_loss(current_q, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network
    update_steps += NUM_ENVS
    if update_steps >= TARGET_UPDATE_FREQ:
        target_net.load_state_dict(policy_net.state_dict())
        update_steps = 0

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

# --- Plot results ---
plt.plot(total_rewards_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title(f"DQN Training on LunarLander-v3 ({NUM_ENVS} envs)")
plt.grid()
plt.show()

torch.save(policy_net.state_dict(), "dqn_lunarlander_parallel.pth")
print("Model saved as dqn_lunarlander_parallel.pth")
