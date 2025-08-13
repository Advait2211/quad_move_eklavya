from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from tqdm import tqdm

# --- Device setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# --- Parallel Environment Setup ---
ENV_ID = "BipedalWalker-v3"
NUM_ENVS = 32         # more parallel envs = faster rollout collection
STEPS_PER_ENV = 256   # shorter rollouts for faster iterations
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
TOTAL_ITERS = 1000

def make_env():
    def thunk():
        return gym.make(ENV_ID)
    return thunk

env = gym.vector.AsyncVectorEnv([make_env() for _ in range(NUM_ENVS)])
obs_size = env.single_observation_space.shape[0]
act_size = env.single_action_space.shape[0]

# --- Actor-Critic ---
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs):
        raise NotImplementedError

    def act(self, obs):
        mean = self.actor(obs)
        dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.1)
        action = dist.sample()
        logprob = dist.log_prob(action).sum(axis=-1)
        value = self.critic(obs)
        return action, logprob, value

    def evaluate(self, obs, actions):
        mean = self.actor(obs)
        dist = torch.distributions.Normal(mean, torch.ones_like(mean) * 0.1)
        logprob = dist.log_prob(actions).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)
        return logprob, entropy, value

# --- Rollout Buffer ---
class RolloutBuffer:
    def __init__(self, steps, num_envs, obs_dim, act_dim):
        self.obs = np.zeros((steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps, num_envs, act_dim), dtype=np.float32)
        self.logprobs = np.zeros((steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps, num_envs), dtype=np.float32)
        self.values = np.zeros((steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((steps, num_envs), dtype=np.float32)
        self.step = 0

    def add(self, obs, actions, logprobs, rewards, dones, values):
        self.obs[self.step] = obs
        self.actions[self.step] = actions
        self.logprobs[self.step] = logprobs
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.values[self.step] = values  # values is now squeezed before passed
        self.step += 1

    def compute_advantages(self, last_value, gamma, lam):
        adv = np.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                nextnonterminal = 1.0 - self.dones[-1]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t+1]
                nextvalues = self.values[t+1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            adv[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
        self.advantages = adv
        self.returns = self.advantages + self.values

    def get(self, minibatch_size):
        steps, num_envs = self.rewards.shape
        batch_size = steps * num_envs
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        obs = self.obs.reshape(batch_size, -1)
        actions = self.actions.reshape(batch_size, -1)
        logprobs = self.logprobs.reshape(batch_size)
        returns = self.returns.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            yield obs[start:end], actions[start:end], logprobs[start:end], returns[start:end], advantages[start:end]

# --- Init ---
agent = ActorCritic(obs_size, act_size).to(DEVICE)
optimizer_policy = optim.Adam(agent.actor.parameters(), lr=POLICY_LR)
optimizer_value = optim.Adam(agent.critic.parameters(), lr=VALUE_LR)

obs, _ = env.reset(seed=None)

# --- Training ---
for iteration in range(TOTAL_ITERS):
    buffer = RolloutBuffer(STEPS_PER_ENV, NUM_ENVS, obs_size, act_size)
    for step in range(STEPS_PER_ENV):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            action, logprob, value = agent.act(obs_t)
        action_np = action.cpu().numpy()
        next_obs, reward, done, trunc, _ = env.step(action_np)
        buffer.add(obs, action_np, logprob.cpu().numpy(), reward, done, value.cpu().numpy().squeeze())
        obs = next_obs

    with torch.no_grad():
        last_value = agent.critic(torch.tensor(obs, dtype=torch.float32, device=DEVICE)).cpu().numpy().squeeze()
    buffer.compute_advantages(last_value, GAMMA, LAMBDA)

    # PPO Update
    for _ in range(NUM_EPOCHS):
        for batch_obs, batch_actions, batch_logprobs, batch_returns, batch_advantages in buffer.get(MINIBATCH_SIZE):
            b_obs = torch.tensor(batch_obs, dtype=torch.float32, device=DEVICE)
            b_actions = torch.tensor(batch_actions, dtype=torch.float32, device=DEVICE)
            b_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32, device=DEVICE)
            b_returns = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
            b_advantages = torch.tensor(batch_advantages, dtype=torch.float32, device=DEVICE)

            new_logprobs, entropy, value = agent.evaluate(b_obs, b_actions)
            ratio = torch.exp(new_logprobs - b_logprobs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * b_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(value.squeeze(), b_returns)
            entropy_loss = -entropy.mean()

            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            loss = policy_loss + VALUE_LOSS_COEF * value_loss + ENTROPY_COEF * entropy_loss
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer_policy.step()
            optimizer_value.step()

    if iteration % 10 == 0:
        print(f"Iter {iteration} done.")

env.close()
