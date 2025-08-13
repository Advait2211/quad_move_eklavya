from tqdm import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from gymnasium.wrappers import ClipAction

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_ID = "BipedalWalker-v3"
NUM_ENVS = 16                  # set to 1 for single-env; >1 for vectorized
TOTAL_ITERS = 2000
STEPS_PER_ENV = 256           # T per env → total steps per iter = NUM_ENVS * STEPS_PER_ENV
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
NUM_EPOCHS = 5
MINIBATCH_SIZE = 8192
VALUE_LOSS_COEF = 1.0
ENTROPY_COEF = 0.01
POLICY_LR = 3e-4
VALUE_LR = 1e-3
MAX_GRAD_NORM = 0.5
EVAL_EPISODES = 10


# -------------------------
# Networks
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std

    def dist(self, x):
        mean, log_std = self.forward(x)
        std = torch.exp(log_std)
        return Normal(mean, std)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.out(x).squeeze(-1)


# -------------------------
# Env helpers
# -------------------------
def make_single_env():
    env = gym.make(ENV_ID)
    env = ClipAction(env)
    return env

def make_vector_env(n):
    # Build SyncVectorEnv of wrapped single envs
    def thunk():
        e = gym.make(ENV_ID)
        e = ClipAction(e)
        return e
    return gym.vector.SyncVectorEnv([thunk for _ in range(n)])


# -------------------------
# Rollout / Trajectory
# -------------------------
@torch.no_grad()
def collect_rollout(env, policy: PolicyNetwork, value_net: ValueNetwork, steps_per_env: int):
    """
    Supports both single and vector envs.
    Returns dict of torch tensors on DEVICE with shapes:
      states:    (N, obs_dim)
      actions:   (N, act_dim)
      rewards:   (N,)
      dones:     (N,)
      values:    (N,)
      logprobs:  (N,)
      next_values_last_step_per_env: (num_envs,) values for bootstrap (not used here since we compute GAE online style)
    """
    vectorized = hasattr(env, "num_envs")
    num_envs = env.num_envs if vectorized else 1

    if vectorized:
        states, _ = env.reset()
    else:
        s, _ = env.reset()
        states = np.expand_dims(s, axis=0)

    obs_dim = states.shape[-1]
    act_dim = env.single_action_space.shape[0] if vectorized else env.action_space.shape[0]

    # Buffers
    states_buf = np.zeros((steps_per_env * num_envs, obs_dim), dtype=np.float32)
    actions_buf = np.zeros((steps_per_env * num_envs, act_dim), dtype=np.float32)
    rewards_buf = np.zeros((steps_per_env * num_envs,), dtype=np.float32)
    dones_buf = np.zeros((steps_per_env * num_envs,), dtype=np.float32)
    values_buf = np.zeros((steps_per_env * num_envs,), dtype=np.float32)
    logprobs_buf = np.zeros((steps_per_env * num_envs,), dtype=np.float32)

    for t in range(steps_per_env):
        # Torchify states
        s_t = torch.tensor(states, dtype=torch.float32, device=DEVICE)
        dist = policy.dist(s_t)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(dim=-1)  # (num_envs,)
        value = value_net(s_t)                        # (num_envs,)

        # Convert actions to numpy with correct shape
        act_np = action.cpu().numpy()
        # Step
        if vectorized:
            next_states, rewards, dones, truncs, infos = env.step(act_np)
            done_flags = np.logical_or(dones, truncs).astype(np.float32)
        else:
            ns, r, d, tr, info = env.step(act_np.squeeze(0))
            next_states = np.expand_dims(ns, axis=0)
            rewards = np.array([r], dtype=np.float32)
            done_flags = np.array([float(d or tr)], dtype=np.float32)

        idx_start = t * num_envs
        idx_end = idx_start + num_envs

        states_buf[idx_start:idx_end] = states
        actions_buf[idx_start:idx_end] = act_np
        rewards_buf[idx_start:idx_end] = rewards.astype(np.float32)
        dones_buf[idx_start:idx_end] = done_flags
        values_buf[idx_start:idx_end] = value.cpu().numpy().astype(np.float32)
        logprobs_buf[idx_start:idx_end] = logprob.cpu().numpy().astype(np.float32)

        # Reset envs that are done
        if vectorized:
            if np.any(done_flags > 0.5):
                # Gymnasium vec API resets automatically on next step; to get fresh obs now:
                done_idx = np.where(done_flags > 0.5)[0]
                for i in done_idx:
                    ns_i, _ = env.envs[i].reset()
                    next_states[i] = ns_i
        else:
            if done_flags[0] > 0.5:
                ns, _ = env.reset()
                next_states[0] = ns

        states = next_states

    # To torch
    traj = {
        "states": torch.tensor(states_buf, dtype=torch.float32, device=DEVICE),
        "actions": torch.tensor(actions_buf, dtype=torch.float32, device=DEVICE),
        "rewards": torch.tensor(rewards_buf, dtype=torch.float32, device=DEVICE),
        "dones": torch.tensor(dones_buf, dtype=torch.float32, device=DEVICE),
        "values": torch.tensor(values_buf, dtype=torch.float32, device=DEVICE),
        "logprobs": torch.tensor(logprobs_buf, dtype=torch.float32, device=DEVICE),
        "num_envs": num_envs,
    }
    return traj


def compute_gae(traj, gamma=0.99, lam=0.95):
    """
    traj fields: states (N, obs), actions (N, act), rewards (N,), dones (N,), values (N,)
    Returns: states, actions, old_logps, returns, advantages (tensors)
    """
    rewards = traj["rewards"].cpu().numpy()
    dones = traj["dones"].cpu().numpy()
    values = traj["values"].cpu().numpy()

    N = len(rewards)
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(N)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * gae * mask
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values

    # Normalize advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std() + 1e-8
    advantages = (advantages - adv_mean) / adv_std

    states = traj["states"]
    actions = traj["actions"]
    old_logps = traj["logprobs"]

    return (
        states,
        actions,
        torch.tensor(old_logps, dtype=torch.float32, device=DEVICE) if not torch.is_tensor(old_logps) else old_logps,
        torch.tensor(returns, dtype=torch.float32, device=DEVICE),
        torch.tensor(advantages, dtype=torch.float32, device=DEVICE),
    )


# -------------------------
# PPO Update
# -------------------------
def ppo_update(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    opt_pi,
    opt_v,
    states,
    actions,
    old_logps,
    returns,
    advantages,
    clip_eps=0.2,
    num_epochs=10,
    minibatch_size=64,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=0.5,
):
    N = states.shape[0]
    last_info = {}

    for _ in range(num_epochs):
        idx = np.arange(N)
        np.random.shuffle(idx)
        for start in range(0, N, minibatch_size):
            mb = idx[start:start + minibatch_size]
            s = states[mb]
            a = actions[mb]
            old_lp = old_logps[mb]
            ret = returns[mb]
            adv = advantages[mb]

            mean, log_std = policy(s)
            dist = Normal(mean, torch.exp(log_std))
            new_logps = dist.log_prob(a).sum(dim=-1)

            ratio = torch.exp(new_logps - old_lp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().sum(dim=-1).mean()

            value_pred = value_net(s)
            value_loss = F.mse_loss(value_pred, ret)

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            opt_pi.zero_grad()
            opt_v.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(value_net.parameters(), max_grad_norm)
            opt_pi.step()
            opt_v.step()

            with torch.no_grad():
                approx_kl = (old_lp - new_logps).mean().item()
                clipfrac = ((ratio > 1.0 + clip_eps) | (ratio < 1.0 - clip_eps)).float().mean().item()

            last_info = {
                "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()),
                "entropy": float(entropy.item()),
                "approx_kl": approx_kl,
                "clipfrac": clipfrac,
            }

    return last_info


# -------------------------
# Train / Evaluate
# -------------------------
def train():
    # Build env
    if NUM_ENVS == 1:
        env = make_single_env()
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        env = make_vector_env(NUM_ENVS)
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]

    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    value_net = ValueNetwork(obs_dim).to(DEVICE)
    opt_pi = torch.optim.Adam(policy.parameters(), lr=POLICY_LR)
    opt_v = torch.optim.Adam(value_net.parameters(), lr=VALUE_LR)

    for it in tqdm(range(TOTAL_ITERS), desc="PPO iters"):
        traj = collect_rollout(env, policy, value_net, STEPS_PER_ENV)
        states, actions, old_logps, returns, advantages = compute_gae(traj, GAMMA, LAMBDA)

        info = ppo_update(
            policy,
            value_net,
            opt_pi,
            opt_v,
            states,
            actions,
            old_logps,
            returns,
            advantages,
            clip_eps=CLIP_EPS,
            num_epochs=NUM_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            value_loss_coef=VALUE_LOSS_COEF,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
        )

        if (it + 1) % 10 == 0:
            print(
                f"Iter {it+1}: "
                f"pi_loss={info.get('policy_loss', 0):.4f}, "
                f"v_loss={info.get('value_loss', 0):.4f}, "
                f"ent={info.get('entropy', 0):.4f}, "
                f"kl={info.get('approx_kl', 0):.6f}, "
                f"clipfrac={info.get('clipfrac', 0):.4f}"
            )

    torch.save(policy.state_dict(), "ppo_bipedal_policy.pth")
    torch.save(value_net.state_dict(), "ppo_bipedal_value_net.pth")
    return env, policy


@torch.no_grad()
def evaluate_policy(policy: PolicyNetwork, episodes=5, render=False):
    env = make_single_env()  # evaluation with single env for simplicity
    total_rewards = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        trunc = False
        ep_r = 0.0
        while not (done or trunc):
            st = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, _ = policy(st)
            action = mean.squeeze(0).cpu().numpy()  # deterministic eval
            # clip action explicitly (also wrapped)
            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            s, r, done, trunc, _ = env.step(action)
            ep_r += float(r)
            if render:
                env.render()
        total_rewards.append(ep_r)
    avg = float(np.mean(total_rewards))
    print(f"Evaluation over {episodes} episodes: Average Reward = {avg:.2f}")
    return avg


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    env, policy = train()
    evaluate_policy(policy, episodes=EVAL_EPISODES, render=False)
