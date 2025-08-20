import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from gymnasium.vector import AsyncVectorEnv
from tqdm import trange
import wandb
import os

# ========================
# CONFIG
# ========================
ENV_ID = "Ant-v5"
NUM_ENVS = 128
STEPS_PER_ENV = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 30
MINIBATCH_SIZE = 65536
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========================
# WANDB SETUP
# ========================

wandb.init(
    project="ppo-go2",
    config={
        "env_id": ENV_ID,
        "num_envs": NUM_ENVS,
        "steps_per_env": STEPS_PER_ENV,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_eps": CLIP_EPS,
        "learning_rate": LR,
        "epochs": EPOCHS,
        "minibatch_size": MINIBATCH_SIZE,
        "device": str(DEVICE),
    }
)

# ========================
# CHECKPOINT SETUP
# ========================


CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, update):
    path = os.path.join(CHECKPOINT_DIR, f"ppo_go2_update_{update}.pth")
    torch.save({
        "update": update,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    wandb.save(path)
    print(f"[Checkpoint] Saved at update {update} → {path}")

def load_checkpoint(model, optimizer, path, device=DEVICE):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded from {path} (Update {checkpoint['update']})")
    return checkpoint["update"]


# ========================
# ENV CREATION
# ========================
def make_env():
    def _init():
        return gym.make(
            ENV_ID,
            xml_file="./unitree_go2/scene.xml",
            forward_reward_weight=2,
            ctrl_cost_weight=0.1,
            contact_cost_weight=0.01,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.45, 0.65),
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.01,
            frame_skip=2,
            max_episode_steps=1000,
            render_mode=None,
        )
    return _init

envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])

# ========================
# MODEL
# ========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 1024
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
# GAE FUNCTION
# ========================
def compute_gae(rewards, values, dones, gamma, lam):
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards, device=DEVICE)
    gae = torch.zeros(num_envs, device=DEVICE)
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

# ========================
# INIT
# ========================
obs, _ = envs.reset()
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

model = ActorCritic(state_dim, action_dim).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

wandb.watch(model, log="all", log_freq=100)

# ========================
# TRAINING LOOP
# ========================
START_UPDATE = 0


for update in trange(START_UPDATE, 40000, desc="PPO Updates"):
    # Buffers (GPU)
    obs_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, state_dim, device=DEVICE)
    act_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, action_dim, device=DEVICE)
    logp_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    rew_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    val_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    done_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)

    for step in range(STEPS_PER_ENV):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mean, std, value = model(obs_t)
            dist = Normal(mean, std)
            action = dist.sample()
            logp = dist.log_prob(action).sum(axis=-1)

        episode_returns = np.zeros(NUM_ENVS, dtype=np.float32)
        completed_returns = []

        next_obs, reward, term, trunc, _ = envs.step(action.cpu().numpy())
        done = term | trunc

        # log new rewards
        episode_returns += reward
        for i, d in enumerate(done):
            if d:
                completed_returns.append(episode_returns[i])
                episode_returns[i] = 0

        # Store
        obs_buf[step] = obs_t
        act_buf[step] = action
        logp_buf[step] = logp
        rew_buf[step] = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        val_buf[step] = value.squeeze()
        done_buf[step] = torch.tensor(done, dtype=torch.float32, device=DEVICE)

        obs = next_obs

    with torch.no_grad():
        next_value = model(torch.tensor(obs, dtype=torch.float32, device=DEVICE))[2].squeeze()
    val_buf = torch.cat([val_buf, next_value.unsqueeze(0)])  # (steps+1, num_envs)

    # Compute GAE
    advantages, returns = compute_gae(rew_buf, val_buf, done_buf, GAMMA, GAE_LAMBDA)

    # Flatten
    obs_tensor = obs_buf.reshape(-1, state_dim)
    act_tensor = act_buf.reshape(-1, action_dim)
    logp_tensor = logp_buf.reshape(-1)
    adv_tensor = advantages.reshape(-1)
    ret_tensor = returns.reshape(-1)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    # PPO update
    total_steps = STEPS_PER_ENV * NUM_ENVS
    for _ in range(EPOCHS):
        idx = torch.randperm(total_steps, device=DEVICE)
        for start in range(0, total_steps, MINIBATCH_SIZE):
            batch_idx = idx[start:start+MINIBATCH_SIZE]

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

    if (update + 1) % 1000 == 0:
        save_checkpoint(model, optimizer, update + 1)

    wandb.log({
        "update": update,
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "total_loss": loss.item(),
        "mean_return": rew_buf.sum(dim=0).mean().item(),
        "mean_advantage": adv_tensor.mean().item(),
        "std_advantage": adv_tensor.std().item(),
    })

    if completed_returns:
        wandb.log({
            "mean_episode_return": np.mean(completed_returns),
            "max_episode_return": np.max(completed_returns),
            "min_episode_return": np.min(completed_returns)
        })
        completed_returns.clear()

# ========================
# EVALUATION
# ========================
eval_env = make_env()()
obs, _ = eval_env.reset()
for _ in range(125):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        mean, _, _ = model(obs_t)
        action = mean.cpu().numpy()[0]
    obs, _, terminated, truncated, _ = eval_env.step(action)
    if terminated or truncated:
        obs, _ = eval_env.reset()
eval_env.close()

# ========================
# SAVE TRAINED MODEL
# ========================
MODEL_PATH = "ppo_go2.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

wandb.save(MODEL_PATH)

