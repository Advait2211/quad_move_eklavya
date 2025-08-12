import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from collections import namedtuple

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
        # TODO: return torch.distributions.Normal (or similar)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)  # std must be positive
        return Normal(mean, std)


    def log_prob(self, dist, action):
        # TODO: compute log π(a|s)
        return dist.log_prob(action).sum(axis=-1)


# 2. Value Network
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        # TODO: define layers
        hidden_dim = 128
        self.layer1 = torch.nn.Linear(state_dim, hidden_dim)
        self.layer2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
    def forward(self, state):
        # TODO: return value estimate
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        value = self.output_layer(x)
        return value.squeeze(-1) 


# 3. Run policy for T timesteps / N trajectories
def collect_trajectories(env, policy, value_net, T):
    # TODO: run env, store (state, action, reward, log_prob, value) per step
    trajectory = []
    state, _ = env.reset()
    for t in range(T):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
        dist = policy.get_distribution(state_tensor)
        action = dist.sample()
        log_prob = policy.log_prob(dist, action)

        value = value_net(state_tensor)

        next_state, reward, done, _, = env.step(action.numpy())
        
        trajectory.append((state, action.numpy(), reward, log_prob.item(), value.item()))

        state = next_state
        if done:
            state, _ = env.reset()
    
    return trajectory

# 4. Estimate advantage function
def compute_advantages(trajectory, value_net, gamma): #monte carlo

    states, actions, rewards, log_probs, values = zip(*trajectory)
    
    values = torch.tensor(values, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    log_probs = torch.stack(log_probs)

    Qvals = []
    Qval = 0
    for reward in reversed(rewards):
        Qval = reward + gamma * Qval
        Qvals.insert(0, Qval)
    Qvals = torch.tensor(Qvals, dtype=torch.float32)
    advantages = Qvals - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, log_probs

# 5. Compute policy gradient g
def compute_policy_gradient(policy, trajectories):
    # TODO: compute gradient of surrogate loss wrt policy parameters

# 6. Conjugate Gradient to solve F^-1 g
def conjugate_gradient(hvp_func, g, max_iter=10):
    # TODO: implement CG algorithm

# 7. Line search with KL constraint
def line_search(policy, full_step, expected_improve_rate, max_backtracks=10):
    # TODO: scale step until KL constraint is satisfied

# 8. Main training loop
def train_trpo(env, policy, value_net, iterations, T, gamma, lam):
    for i in range(iterations):
        # Step 1: collect trajectories
        trajectories = collect_trajectories(env, policy, T)
        
        # Step 2: estimate advantage
        advantages = compute_advantages(trajectories, value_net, gamma, lam)
        
        # Step 3: compute policy gradient
        g = compute_policy_gradient(policy, trajectories)
        
        # Step 4: use CG to get step direction
        step_dir = conjugate_gradient(hvp_func, g)
        
        # Step 5: do line search to update policy
        line_search(policy, step_dir, expected_improve_rate=None)
        
        # Step 6: update value function
        # TODO: fit value_net to returns