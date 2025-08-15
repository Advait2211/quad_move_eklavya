import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from gymnasium.wrappers import ClipAction

device = torch.device("gpu")  # change to "cuda" if available

# -------------------------
# Your networks (unchanged)
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__() 
        hidden_dim = 128
        self.layer1 = torch.nn.Linear(state_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        return mean, log_std

    def get_distribution(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def get_distribution_params(self, states):
        mean, log_std = self.forward(states)
        log_std = torch.clamp(log_std, -20, 2)  # prevents overflow
        return mean, log_std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        hidden_dim = 128
        self.layer1 = torch.nn.Linear(state_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        value = self.output_layer(x)
        return value.squeeze(-1)


# -------------------------
# 3. Collect trajectories
# -------------------------
def collect_trajectories(env, policy: PolicyNetwork, value_net: ValueNetwork, T):
    traj = []
    state, _ = env.reset()
    for t in range(T):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, log_std = policy.forward(state_tensor)
            log_std = torch.clamp(log_std, -20.0, 2.0)
            std = torch.exp(log_std)

            dist = Normal(mean, std)
            action_tensor = dist.sample()
            logp_tensor = dist.log_prob(action_tensor).sum(axis=-1)
            value_tensor = value_net(state_tensor)

        action = action_tensor.squeeze(0).cpu().numpy()

        if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        logp = logp_tensor.item()
        value = value_tensor.item()

        next_state, reward, done, truncated, info = env.step(action)
        traj.append((state, action, reward, logp, value))

        state = next_state
        if done or truncated:
            state, _ = env.reset()

    return traj


# -------------------------
# 4. Compute advantages (GAE) + returns
# -------------------------
def compute_advantages(trajectory, value_net: ValueNetwork, gamma: float, lam: float):
    states, actions, rewards, old_logps, values = zip(*trajectory)
    rewards = np.array(rewards, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextvalue = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * nextvalue - values[t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages[t] = lastgaelam

    returns = advantages + values

    adv_mean = advantages.mean()
    adv_std = advantages.std() if advantages.std() > 0 else 1.0
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    states_t = torch.tensor(np.array(states, dtype=np.float32), dtype=torch.float32).to(device)
    actions_t = torch.tensor(np.array(actions, dtype=np.float32), dtype=torch.float32).to(device)
    old_logps_t = torch.tensor(np.array(old_logps, dtype=np.float32), dtype=torch.float32).to(device)
    returns_t = torch.tensor(returns, dtype=torch.float32).to(device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32).to(device)

    return states_t, actions_t, old_logps_t, returns_t, advantages_t


# -------------------------
# 5. PPO mini-batch update
# -------------------------
def ppo_update(
    policy: PolicyNetwork,
    value_net: ValueNetwork,
    optimizer_policy,
    optimizer_value,
    states,
    actions,
    old_logps,
    returns,
    advantages,
    clip_eps=0.2,
    num_epochs=10,
    mini_batch_size=64,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    max_grad_norm=0.5,
):
    N = states.shape[0]
    for epoch in range(num_epochs):
        idxs = np.arange(N)
        np.random.shuffle(idxs)
        for start in range(0, N, mini_batch_size):
            mb_idx = idxs[start : start + mini_batch_size]
            mb_states = states[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logps = old_logps[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            mean, log_std = policy.get_distribution_params(mb_states)
            dist = Normal(mean, torch.exp(log_std))
            new_logps = dist.log_prob(mb_actions).sum(axis=-1)
            ratio = torch.exp(new_logps - mb_old_logps)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = dist.entropy().sum(axis=-1).mean()

            value_pred = value_net(mb_states).squeeze(-1)
            value_loss = F.mse_loss(value_pred, mb_returns)

            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            optimizer_policy.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer_policy.step()

    with torch.no_grad():
        mean, log_std = policy.get_distribution_params(states)
        dist = Normal(mean, torch.exp(log_std))
        new_logps_all = dist.log_prob(actions).sum(axis=-1)
        ratios_all = torch.exp(new_logps_all - old_logps)
        approx_kl = (old_logps - new_logps_all).mean().item()
        clipfrac = ((ratios_all > 1.0 + clip_eps) | (ratios_all < 1.0 - clip_eps)).float().mean().item()

    return {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
    }


# -------------------------
# 6. Training loop for PPO
# -------------------------
def train_ppo(
    env,
    policy,
    value_net,
    iterations,
    T,
    gamma=0.99,
    lam=0.95,
    clip_eps=0.2,
    num_epochs=10,
    mini_batch_size=64,
    value_loss_coef=1.0,
    entropy_coef=0.0,
    policy_lr=3e-4,
    value_lr=1e-3,
    max_grad_norm=0.5,
):
    policy.to(device)
    value_net.to(device)
    optimizer_policy = torch.optim.Adam(policy.parameters(), lr=policy_lr)
    optimizer_value = torch.optim.Adam(value_net.parameters(), lr=value_lr)

    for it in range(iterations):
        traj = collect_trajectories(env, policy, value_net, T)
        states, actions, old_logps, returns, advantages = compute_advantages(traj, value_net, gamma, lam)
        info = ppo_update(
            policy,
            value_net,
            optimizer_policy,
            optimizer_value,
            states,
            actions,
            old_logps,
            returns,
            advantages,
            clip_eps=clip_eps,
            num_epochs=num_epochs,
            mini_batch_size=mini_batch_size,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )
        print(
            f"Iter {it}: policy_loss={info['policy_loss']:.4f}, "
            f"value_loss={info['value_loss']:.4f}, entropy={info['entropy']:.4f}, "
            f"approx_kl={info['approx_kl']:.6f}, clipfrac={info['clipfrac']:.4f}"
        )

    torch.save(policy.state_dict(), "ppo_bipedal_policy.pth")
    torch.save(value_net.state_dict(), "ppo_bipedal_value_net.pth")

    return policy, value_net


# -------------------------
# 7. Evaluation function
# -------------------------
def evaluate_policy(env, policy, episodes=10, render=False):
    policy.eval()
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = policy.forward(state_tensor)
                action = mean.squeeze(0).cpu().numpy()  # deterministic action (no sampling)
            if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
        total_rewards.append(ep_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Evaluation over {episodes} episodes: Average Reward = {avg_reward:.2f}")
    return avg_reward


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    env = gym.make("BipedalWalker-v3")
    env = ClipAction(env)  # clip actions to valid range

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = PolicyNetwork(obs_dim, act_dim)
    value_net = ValueNetwork(obs_dim)

    # Train PPO
    policy, value_net = train_ppo(
        env,
        policy,
        value_net,
        iterations=200,
        T=2048,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        num_epochs=10,
        mini_batch_size=64,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        policy_lr=3e-4,
        value_lr=1e-3,
        max_grad_norm=0.5,
    )

    # Evaluate trained policy (change render=True to watch)
    evaluate_policy(env, policy, episodes=10, render=False)
