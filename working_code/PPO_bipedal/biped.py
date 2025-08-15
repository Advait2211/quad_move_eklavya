import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import trange
import numpy as np

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
MINIBATCH_SIZE = 4096
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

# ========================
# GAE ADVANTAGE FUNCTION (GPU)
# ========================
def compute_gae(rewards, values, dones, gamma, lam):
    advantages = torch.zeros_like(rewards, device=DEVICE)
    gae = torch.zeros(rewards.size(1), device=DEVICE)  # per env
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

# ========================
# MAIN TRAIN LOOP
# ========================
from gymnasium.vector import SyncVectorEnv

def make_env():
    return lambda: gym.make(ENV_ID, render_mode=None)

envs = SyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs, _ = envs.reset()

state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

model = ActorCritic(state_dim, action_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# print("CUDA available:", torch.cuda.is_available())
# print("Model device:", next(model.parameters()).device)

for update in trange(4000, desc="PPO Updates"):
    obs_buf = torch.zeros((STEPS_PER_ENV, NUM_ENVS, state_dim), device=DEVICE)
    act_buf = torch.zeros((STEPS_PER_ENV, NUM_ENVS, action_dim), device=DEVICE)
    logp_buf = torch.zeros((STEPS_PER_ENV, NUM_ENVS), device=DEVICE)
    rew_buf = torch.zeros((STEPS_PER_ENV, NUM_ENVS), device=DEVICE)
    val_buf = torch.zeros((STEPS_PER_ENV + 1, NUM_ENVS), device=DEVICE)
    done_buf = torch.zeros((STEPS_PER_ENV, NUM_ENVS), device=DEVICE)

    # --- 1. COLLECT TRAJECTORIES ---
    for step in range(STEPS_PER_ENV):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        obs_t = torch.from_numpy(obs).float().to(DEVICE, non_blocking=True)
        with torch.no_grad():
            mean, std, value = model(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)

        next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
        done = term | trunc

        obs_buf[step] = obs_t
        act_buf[step] = action
        logp_buf[step] = logp
        rew_buf[step] = torch.from_numpy(reward).float().to(DEVICE, non_blocking=True)
        val_buf[step] = value.squeeze()
        done_buf[step] = torch.from_numpy(done.astype(np.float32)).to(DEVICE, non_blocking=True)

        obs = next_obs

    # --- 2. GAE + RETURNS ---
    with torch.no_grad():
        next_value = model(torch.tensor(obs, dtype=torch.float32, device=DEVICE))[2].squeeze()
    val_buf[-1] = next_value

    advantages, returns = compute_gae(rew_buf, val_buf, done_buf, GAMMA, GAE_LAMBDA)

    obs_tensor = obs_buf.reshape(-1, state_dim)
    act_tensor = act_buf.reshape(-1, action_dim)
    logp_tensor = logp_buf.reshape(-1)
    adv_tensor = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)
    ret_tensor = returns.reshape(-1)

    # --- 3. PPO UPDATE ---
    total_steps = obs_tensor.size(0)
    for _ in range(EPOCHS):
        idx = torch.randperm(total_steps, device=DEVICE)
        for start in range(0, total_steps, MINIBATCH_SIZE):
            batch_idx = idx[start:start + MINIBATCH_SIZE]

            mean, std, value = model(obs_tensor[batch_idx])
            dist = Normal(mean, std)
            new_logp = dist.log_prob(act_tensor[batch_idx]).sum(axis=-1)
            ratio = (new_logp - logp_tensor[batch_idx]).exp()

            surr1 = ratio * adv_tensor[batch_idx]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_tensor[batch_idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = ((value.squeeze() - ret_tensor[batch_idx]) ** 2).mean()
            entropy_loss = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

envs.close()


# ========================
# SAVE TRAINED MODEL
# ========================
MODEL_PATH = "ppo_bipedalwalker.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
