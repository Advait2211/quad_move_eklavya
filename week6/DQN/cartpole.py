# dqn.py
import random
import collections
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import time
import os

# -------------------------
# Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # store as numpy arrays for compactness
        self.buffer.append((np.array(state, copy=False),
                            int(action),
                            float(reward),
                            np.array(next_state, copy=False),
                            bool(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# -------------------------
# Q-Network (MLP for vector states)
# -------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# DQN Agent
# -------------------------
class DQNAgent:
    def __init__(
        self,
        env,
        state_dim,
        action_dim,
        replay_capacity=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        sync_freq=1000,
        start_learning=1000,
        eps_start=1.0,
        eps_final=0.01,
        eps_decay=50000,
        device=None
    ):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_freq = sync_freq
        self.start_learning = start_learning

        # epsilon schedule
        self.eps_start = eps_start
        self.eps_final = eps_final
        self.eps_decay = eps_decay
        self.total_steps = 0

    def select_action(self, state, eval_mode=False):
        eps = self.epsilon()
        if eval_mode or random.random() > eps:
            state_v = torch.tensor(np.array(state, dtype=np.float32), device=self.device).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q_net(state_v)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            action = self.env.action_space.sample()
        return action

    def epsilon(self):
        # linear decay
        t = min(self.total_steps, self.eps_decay)
        return self.eps_final + (self.eps_start - self.eps_final) * (1 - t / self.eps_decay)

    def learn_step(self):
        if len(self.replay) < max(self.batch_size, self.start_learning):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states_v = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_v = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_v = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_v = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # current Q values for taken actions
        q_values = self.q_net(states_v).gather(1, actions_v)

        # compute target using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states_v)
            max_next_q_values, _ = torch.max(next_q_values, dim=1, keepdim=True)
            target = rewards_v + self.gamma * (1 - dones_v) * max_next_q_values

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # optional: clip gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_if_needed(self):
        if self.total_steps % self.sync_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())


# -------------------------
# Training loop
# -------------------------
def train(env_name="CartPole-v1",
          max_episodes=1000,
          max_steps_per_episode=1000,
          replay_capacity=100000,
          batch_size=64,
          gamma=0.99,
          lr=1e-3,
          sync_freq=1000,
          start_learning=1000,
          eps_start=1.0,
          eps_final=0.01,
          eps_decay=50000,
          save_path="dqn_checkpoint.pth"):

    env = gym.make(env_name)
    # if using gymnasium replace the above with: env = gym.make(env_name, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        env,
        state_dim,
        action_dim,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        sync_freq=sync_freq,
        start_learning=start_learning,
        eps_start=eps_start,
        eps_final=eps_final,
        eps_decay=eps_decay
    )

    episode_rewards = []
    start_time = time.time()
    global_step = 0

    for ep in range(1, max_episodes + 1):
        state = env.reset()
        # For gymnasium, reset returns (obs, info)
        if isinstance(state, tuple):
            state = state[0]

        ep_reward = 0
        for t in range(max_steps_per_episode):
            action = agent.select_action(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                # some gym versions return (next_state, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            agent.replay.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.total_steps += 1
            global_step += 1

            loss = agent.learn_step()
            agent.update_target_if_needed()

            if done:
                break

        episode_rewards.append(ep_reward)
        # logging
        if ep % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            print(f"Episode {ep} | Steps {global_step} | AvgReward(last10) {avg_last_10:.2f} | Eps {agent.epsilon():.3f} | Replay {len(agent.replay)} | Loss {loss_str}")

        # save checkpoint
        if ep % 100 == 0:
            agent.save(save_path)
    total_time = time.time() - start_time
    print("Training finished. Total time:", total_time)
    agent.save(save_path)
    env.close()
    return agent, episode_rewards


if __name__ == "__main__":
    # Example run; change env_name to "LunarLander-v2" or "LunarLander-v3" if available
    trained_agent, rewards = train(env_name="CartPole-v1", max_episodes=10000)
