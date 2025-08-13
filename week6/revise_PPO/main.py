import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange

# ========================
# CONFIG
# ========================
ENV_ID = "BipedalWalker-v3"
NUM_ENVS = 8
STEPS_PER_ENV = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 10
MINIBATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# NETWORKS
# ========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 64
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        return mean, std, value

def make_env():
    return gym.make(ENV_ID)

# ========================
# GAE ADVANTAGE FUNCTION
# ========================
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

# ========================
# MAIN TRAIN LOOP
# ========================
envs = gym.vector.make(ENV_ID, num_envs=NUM_ENVS)
obs, _ = envs.reset()

state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

model = ActorCritic(state_dim, action_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for update in trange(1000, desc="PPO Updates"):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

    # --- 1. COLLECT TRAJECTORIES ---
    for step in range(STEPS_PER_ENV):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            mean, std, value = model(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)
        
        next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
        done = term | trunc

        obs_buf.append(obs_t.cpu())
        act_buf.append(action.cpu())
        logp_buf.append(logp.cpu())
        rew_buf.append(torch.tensor(reward, dtype=torch.float32))
        val_buf.append(value.squeeze().cpu())
        done_buf.append(torch.tensor(done, dtype=torch.float32))

        obs = next_obs

    # --- 2. GAE + RETURNS ---
    with torch.no_grad():
        next_value = model(torch.tensor(obs, dtype=torch.float32).to(DEVICE))[2].squeeze().cpu()
    val_buf.append(next_value)
    advantages, returns = compute_gae(
        torch.stack(rew_buf),
        torch.stack(val_buf),
        torch.stack(done_buf),
        GAMMA, GAE_LAMBDA
    )

    # Flatten
    obs_tensor = torch.cat(obs_buf)
    act_tensor = torch.cat(act_buf)
    logp_tensor = torch.cat(logp_buf)
    adv_tensor = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    ret_tensor = returns

    # --- 3. PPO UPDATE ---
    for _ in range(EPOCHS):
        idx = np.arange(len(obs_tensor))
        np.random.shuffle(idx)
        for start in range(0, len(idx), MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            batch_idx = idx[start:end]

            mean, std, value = model(obs_tensor[batch_idx].to(DEVICE))
            dist = Normal(mean, std)
            new_logp = dist.log_prob(act_tensor[batch_idx].to(DEVICE)).sum(axis=-1)
            ratio = (new_logp - logp_tensor[batch_idx].to(DEVICE)).exp()

            # PPO Clipped Objective
            surr1 = ratio * adv_tensor[batch_idx].to(DEVICE)
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_tensor[batch_idx].to(DEVICE)
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value.squeeze() - ret_tensor[batch_idx].to(DEVICE)) ** 2).mean()
            entropy_loss = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

envs.close()
