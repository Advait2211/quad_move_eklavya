import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv
from go2_env import Go2Env

# ========================
# CONFIG - INCREASED FOR LONGER TRAINING
# ========================
NUM_ENVS = 8  # Increased from 4
STEPS_PER_ENV = 2048  # Increased from 128
MAX_UPDATES = 50_000  # Increased from 10,000 (5x longer)
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 10
MINIBATCH_SIZE = 8192
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")  # Debug: Check if GPU is detected

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")

EPS = 1e-6
GRAD_CLIP = 0.5

ENTROPY_START = 0.02
ENTROPY_END = 0.001

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def make_env():
    def _init():
        return Go2Env(xml_path="./unitree_go2/scene.xml", render_mode=None)
    return _init

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 512
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
        # Ensure input is on correct device
        if not x.is_cuda and torch.cuda.is_available():
            x = x.cuda()
        value = self.critic(x)
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        return mean, std, value

def compute_gae(rewards, values, dones_term, gamma, lam):
    steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards, device=DEVICE)
    gae = torch.zeros(num_envs, device=DEVICE)
    for t in reversed(range(steps)):
        mask = 1.0 - dones_term[t]
        delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    returns = advantages + values[:-1]
    return advantages, returns

def atanh(x):
    x = x.clamp(-1 + 1e-6, 1 - 1e-6)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))

def main():
    # Set thread settings
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    # Vector envs
    envs = AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)], context="spawn")

    obs, _ = envs.reset()
    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Track best performance
    best_mean_return = -float('inf')

    for update in tqdm(range(MAX_UPDATES), desc="PPO Updates"):
        # ---- Rollout ----
        obs_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, state_dim, device=DEVICE)  # On GPU
        act_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, action_dim, device=DEVICE)
        logp_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
        rew_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
        val_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
        term_buf = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)

        ep_ret = np.zeros(NUM_ENVS, dtype=np.float32)
        finished_returns = []

        for t in range(STEPS_PER_ENV):
            # Move obs to GPU immediately
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            
            with torch.no_grad():
                mean, std, value = model(obs_t)
                dist = Normal(mean, std)
                raw_action = dist.sample()
                act = torch.tanh(raw_action)
                logp_raw = dist.log_prob(raw_action).sum(dim=-1)
                logp = logp_raw - torch.log(1 - act.pow(2) + EPS).sum(dim=-1)

            # Move action back to CPU for environment
            next_obs, reward, term, trunc, _ = envs.step(act.cpu().numpy())
            done = term | trunc

            ep_ret += reward
            for i, d in enumerate(done):
                if d:
                    finished_returns.append(ep_ret[i])
                    ep_ret[i] = 0.0

            # Store on GPU
            obs_buf[t] = obs_t
            act_buf[t] = act
            logp_buf[t] = logp
            rew_buf[t] = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
            val_buf[t] = value.squeeze()
            term_buf[t] = torch.tensor(term.astype(np.float32), device=DEVICE)

            obs = next_obs

        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
            _, _, next_value = model(obs_tensor)
            next_value = next_value.squeeze()
        val_buf = torch.cat([val_buf, next_value.unsqueeze(0)], dim=0)

        adv, ret = compute_gae(rew_buf, val_buf, term_buf, GAMMA, GAE_LAMBDA)

        obs_tensor = obs_buf.reshape(-1, state_dim)
        act_tensor = act_buf.reshape(-1, action_dim)
        logp_tensor = logp_buf.reshape(-1)
        adv_tensor = adv.reshape(-1)
        ret_tensor = ret.reshape(-1)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # ---- PPO updates ----
        frac = update / MAX_UPDATES
        entropy_coef = ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)
        current_lr = LR * (1 - frac)
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        total_steps = STEPS_PER_ENV * NUM_ENVS
        for _ in range(EPOCHS):
            idx = torch.randperm(total_steps, device=DEVICE)
            for s in range(0, total_steps, MINIBATCH_SIZE):
                bi = idx[s:s+MINIBATCH_SIZE]
                b_obs = obs_tensor[bi]
                b_act = act_tensor[bi]
                b_old = logp_tensor[bi]
                b_adv = adv_tensor[bi]
                b_ret = ret_tensor[bi]

                mean, std, value = model(b_obs)
                dist = Normal(mean, std)
                pre = atanh(b_act)
                new_raw = dist.log_prob(pre).sum(dim=-1)
                new_logp = new_raw - torch.log(1 - b_act.pow(2) + EPS).sum(dim=-1)

                ratio = (new_logp - b_old).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v = value.squeeze()
                v_clipped = v + (v - b_ret).clamp(-CLIP_EPS, CLIP_EPS)
                value_loss = torch.max((v - b_ret) ** 2, (v_clipped - b_ret) ** 2).mean()

                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

        if finished_returns:
            mean_return = np.mean(finished_returns)
            # print(f"[Update {update}] MeanRet {mean_return:.2f} | Ent {entropy_coef:.4f} | LR {current_lr:.6f}")
            
            # Save best model
            if mean_return > best_mean_return:
                best_mean_return = mean_return
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "ppo_go2_best.pth"))
                print(f"💾 New best model saved! Return: {best_mean_return:.2f}")

        if (update + 1) % 1000 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ppo_go2_{update+1}.pth"))

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    main()