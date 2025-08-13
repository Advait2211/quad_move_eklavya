import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# Policy and Value Networks
# ===============================
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_layer = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.mean_layer(x)
        std = torch.exp(self.log_std)
        return mean, std

    def act(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_layer = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.value_layer(x)

# ===============================
# Trajectory Collection
# ===============================
def collect_trajectories(env, policy, value_net, steps_per_env):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    observations, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

    for _ in range(steps_per_env):
        with torch.no_grad():
            action, log_prob = policy.act(obs)
            value = value_net(obs).squeeze(-1)

        next_obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
        done_flag = np.logical_or(done, truncated)

        observations.append(obs)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done_flag, dtype=torch.float32, device=device))
        values.append(value)

        obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    # Stack tensors
    observations = torch.stack(observations)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    values = torch.stack(values)

    return observations, actions, log_probs, rewards, dones, values

# ===============================
# GAE-Lambda
# ===============================
def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards, device=device)
    gae = 0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages, returns

# ===============================
# PPO Update
# ===============================
def ppo_update(policy, value_net, optimizer_policy, optimizer_value,
               observations, actions, log_probs_old, returns, advantages,
               clip_ratio=0.2, value_loss_coef=0.5, entropy_coef=0.01, epochs=10, batch_size=64):
    
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = observations.size(0)
    for _ in range(epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            obs_b = observations[batch_idx]
            actions_b = actions[batch_idx]
            old_log_probs_b = log_probs_old[batch_idx]
            returns_b = returns[batch_idx]
            adv_b = advantages[batch_idx]

            mean, std = policy(obs_b)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions_b).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratio = torch.exp(log_probs - old_log_probs_b)
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            value_pred = value_net(obs_b).squeeze(-1)
            value_loss = (returns_b - value_pred).pow(2).mean()

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
            loss.backward()
            optimizer_policy.step()
            optimizer_value.step()

# ===============================
# Training Loop
# ===============================
def train_ppo(env_name="BipedalWalker-v3", total_iters=2000, num_envs=8, steps_per_env=2048, gamma=0.99, lam=0.95):
    env = gym.vector.AsyncVectorEnv(
        [lambda: gym.make(env_name) for _ in range(num_envs)],
        auto_reset=True
    )

    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim).to(device)
    value_net = ValueNetwork(obs_dim).to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

    for _ in tqdm(range(total_iters), desc="PPO iters"):
        observations, actions, log_probs_old, rewards, dones, values = collect_trajectories(env, policy, value_net, steps_per_env)

        with torch.no_grad():
            next_value = value_net(observations[-1]).squeeze(-1)

        advantages, returns = compute_gae(rewards, dones, values, next_value, gamma, lam)

        observations = observations.reshape(-1, obs_dim)
        actions = actions.reshape(-1, act_dim)
        log_probs_old = log_probs_old.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)

        ppo_update(policy, value_net, optimizer_policy, optimizer_value,
                   observations, actions, log_probs_old, returns, advantages)

    env.close()
    return policy, value_net

if __name__ == "__main__":
    policy, value_net = train_ppo()
