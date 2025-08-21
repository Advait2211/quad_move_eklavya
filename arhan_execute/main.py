import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm
import wandb
import os

# ========================
# CONFIG
# ========================
ENV_ID = "Ant-v5"
NUM_ENVS = 128
STEPS_PER_ENV = 512
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 30
MINIBATCH_SIZE = 16384
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-6
GRAD_CLIP = 0.5

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
    torch.save(
        {
            "update": update,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )
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
            max_episode_steps=500,
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
        hidden = 256
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.5)
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
# UTIL: stable atanh
# ========================
def atanh(x):
    # clamp inside (-1+eps, 1-eps) then atanh
    x = x.clamp(-1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

# ========================
# GAE FUNCTION
# ========================
def compute_gae(rewards, values, dones_term, gamma, lam):
    # rewards: (steps, envs)
    # values: (steps+1, envs)
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards, device=DEVICE)
    gae = torch.zeros(num_envs, device=DEVICE)
    for t in reversed(range(steps)):
        # dones_term: 1 if env terminated (no bootstrap), 0 else
        mask = 1.0 - dones_term[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
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
wandb.watch(model, log="all")

# ========================
# TRAIN LOOP
# ========================
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# ========================
# TRAIN LOOP
# ========================
START_UPDATE = 0
MAX_UPDATES = 40_000

for update in tqdm(range(START_UPDATE, MAX_UPDATES), desc="PPO Updates"):
    frac = 1.0 - (update / MAX_UPDATES)
    # Buffers
    obs_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, state_dim, device=DEVICE)
    act_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, action_dim, device=DEVICE)
    logp_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    rew_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    val_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    term_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)

    running_returns = np.zeros(NUM_ENVS, dtype=np.float32)
    episode_returns = []

    for step in range(STEPS_PER_ENV):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            mean, std, value = model(obs_t)
            dist = Normal(mean, std)
            raw_action = dist.sample()
            squashed_action = torch.tanh(raw_action)
            logp_raw = dist.log_prob(raw_action).sum(dim=-1)
            logp = logp_raw - torch.log(1 - squashed_action.pow(2) + EPS).sum(dim=-1)

        next_obs, reward, term, trunc, _ = envs.step(squashed_action.cpu().numpy())
        done_any = term | trunc

        running_returns += reward
        for i, d in enumerate(done_any):
            if d:
                episode_returns.append(running_returns[i])
                running_returns[i] = 0.0

        done_term_mask = term.astype(np.float32)

        obs_buf[step]  = obs_t
        act_buf[step]  = squashed_action
        logp_buf[step] = logp
        rew_buf[step]  = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        val_buf[step]  = value.squeeze()
        term_buf[step] = torch.tensor(done_term_mask, dtype=torch.float32, device=DEVICE)

        obs = next_obs

    # bootstrap last value
    with torch.no_grad():
        _, _, next_value = model(torch.tensor(obs, dtype=torch.float32, device=DEVICE))
        next_value = next_value.squeeze()
    val_buf = torch.cat([val_buf, next_value.unsqueeze(0)], dim=0)

    advantages, returns = compute_gae(rew_buf, val_buf, term_buf, GAMMA, GAE_LAMBDA)

    obs_tensor = obs_buf.reshape(-1, state_dim)
    act_tensor = act_buf.reshape(-1, action_dim)
    logp_tensor = logp_buf.reshape(-1)
    adv_tensor = advantages.reshape(-1)
    ret_tensor = returns.reshape(-1)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    # PPO updates
    total_steps = STEPS_PER_ENV * NUM_ENVS
    for _ in range(EPOCHS):
        idx = torch.randperm(total_steps, device=DEVICE)
        for start in range(0, total_steps, MINIBATCH_SIZE):
            batch_idx = idx[start:start+MINIBATCH_SIZE]
            batch_obs = obs_tensor[batch_idx]
            batch_acts = act_tensor[batch_idx]
            batch_old_logp = logp_tensor[batch_idx]
            batch_adv = adv_tensor[batch_idx]
            batch_ret = ret_tensor[batch_idx]

            with autocast():   # <-- AMP forward
                mean, std, value = model(batch_obs)
                dist = Normal(mean, std)
                pre_squash = atanh(batch_acts)
                new_logp_raw = dist.log_prob(pre_squash).sum(dim=-1)
                new_logp = new_logp_raw - torch.log(1 - batch_acts.pow(2) + EPS).sum(dim=-1)

                ratio = (new_logp - batch_old_logp).exp()
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                val_batch = val_buf[:-1].reshape(-1)[batch_idx]
                value_clipped = val_batch + (value.squeeze() - val_batch).clamp(-CLIP_EPS, CLIP_EPS)
                value_loss = torch.max(
                    (value.squeeze() - batch_ret) ** 2,
                    (value_clipped - batch_ret) ** 2
                ).mean()

                entropy = dist.entropy().mean()
                initial_entropy_coef = 0.02
                final_entropy_coef = 0.005
                entropy_coef = final_entropy_coef + (initial_entropy_coef - final_entropy_coef) * frac
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

    # === LR schedule (linear decay) ==
    for g in optimizer.param_groups:
        g["lr"] = LR * frac

    # Save checkpoint
    if (update + 1) % 1000 == 0:
        save_checkpoint(model, optimizer, update + 1)

    # wandb logs
    wandb.log({
        "update": update,
        "lr": optimizer.param_groups[0]["lr"],   # log current LR
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy_loss": entropy.item(),
        "total_loss": loss.item(),
        "mean_rollout_return": rew_buf.sum(dim=0).mean().item(),
        "std_advantage": adv_tensor.std().item(),
    })

    if episode_returns:
        wandb.log({
            "mean_episode_return": float(np.mean(episode_returns)),
            "max_episode_return": float(np.max(episode_returns)),
            "min_episode_return": float(np.min(episode_returns)),
        })
        episode_returns.clear()

# ========================
# EVAL
# ========================
eval_env = make_env()()
NUM_EVAL_EPISODES = 5
for ep in range(NUM_EVAL_EPISODES):
    obs, _ = eval_env.reset()
    done = False
    ep_return = 0.0
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = model(obs_t)
            action = torch.tanh(mean).cpu().numpy()[0]   # use deterministic mean (squashed)
        obs, reward, term, trunc, _ = eval_env.step(action)
        ep_return += reward
        done = bool(term or trunc)
    print(f"Eval Episode {ep + 1}: Return = {ep_return}")
eval_env.close()

# ========================
# SAVE MODEL
# ========================
MODEL_PATH = "ppo_go2.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
wandb.save(MODEL_PATH)
