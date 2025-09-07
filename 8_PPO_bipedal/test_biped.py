import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# ========================
# CONFIG
# ========================
ENV_ID = "BipedalWalker-v3"
MODEL_PATH = "ppo_bipedalwalker.pth"
EPISODES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# ACTOR-CRITIC NETWORK (same as training)
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
# LOAD ENVIRONMENT & MODEL
# ========================
env = gym.make(ENV_ID, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

model = ActorCritic(state_dim, action_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ========================
# EVALUATION LOOP
# ========================
total_rewards = []

for ep in range(EPISODES):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, std, _ = model(obs_tensor)
            dist = Normal(mean, std)
            action = mean  # deterministic (no sampling) for evaluation

        obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        done = terminated or truncated
        ep_reward += reward

    total_rewards.append(ep_reward)
    print(f"Episode {ep+1}: Reward = {ep_reward:.2f}")

env.close()

print(f"\nAverage Reward over {EPISODES} episodes: {np.mean(total_rewards):.2f}")
