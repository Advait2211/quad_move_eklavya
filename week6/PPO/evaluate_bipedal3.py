# evaluate_bipedal.py
import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import ClipAction

# -------------------------
# Config
# -------------------------
DEVICE = torch.device("cuda")  # change to "cuda" if available
ENV_ID = "BipedalWalker-v3"
POLICY_PATH = "ppo_bipedal_policy.pth"
EPISODES = 100  # number of evaluation episodes


# -------------------------
# Policy Network (must match training definition)
# -------------------------
class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, hidden)
        self.mean = torch.nn.Linear(hidden, action_dim)
        self.log_std = torch.nn.Linear(hidden, action_dim)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std


def make_single_env():
    env = gym.make(ENV_ID, render_mode="human")  # human render for visual evaluation
    env = ClipAction(env)
    return env


# -------------------------
# Evaluation
# -------------------------
@torch.no_grad()
def evaluate_policy(policy: PolicyNetwork, episodes=5):
    env = make_single_env()
    total_rewards = []

    for ep in range(episodes):
        s, _ = env.reset()
        done, trunc = False, False
        ep_r = 0.0
        while not (done or trunc):
            st = torch.tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            mean, _ = policy(st)
            action = mean.squeeze(0).cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            s, r, done, trunc, _ = env.step(action)
            ep_r += r
        print(f"Episode {ep + 1}: Reward = {ep_r:.2f}")
        total_rewards.append(ep_r)

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {episodes} episodes: {avg_reward:.2f}")
    env.close()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # Create env to get obs/action dimensions
    temp_env = make_single_env()
    obs_dim = temp_env.observation_space.shape[0]
    act_dim = temp_env.action_space.shape[0]
    temp_env.close()

    # Load policy
    policy = PolicyNetwork(obs_dim, act_dim).to(DEVICE)
    policy.load_state_dict(torch.load(POLICY_PATH, map_location=DEVICE))
    policy.eval()

    evaluate_policy(policy, episodes=EPISODES)
