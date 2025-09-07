import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
import torch
from torch.distributions import Normal
import torch.nn as nn
from tqdm import tqdm

# Import your custom environment
from go2_env import Go2Env

# === PPOAgent definition (as provided) ===
class PPOAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes=(64, 64),
        clip_eps=0.2,
        gamma=0.99,
        lam=0.95,
        lr=3e-4,
        epochs=10,
        minibatches=4,
        initial_entropy_coeff = 0.01,
        final_entropy_coeff = 0.00005,
        value_coeff=0.5,
        max_grad_norm=0.5,
        device=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatches = minibatches
        self.entropy_coeff = initial_entropy_coeff
        self.initial_entropy_coef = initial_entropy_coeff
        self.final_entropy_coef = final_entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cpu")

        # policy network
        layers = []
        last_size = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(last_size, size), nn.Tanh()]
            last_size = size
        layers += [nn.Linear(last_size, act_dim)]
        self.policy_mean = nn.Sequential(*layers)
        self.policy_logstd = nn.Parameter(torch.zeros(1, act_dim))

        # value network
        v_layers = []
        last_size = obs_dim
        for size in hidden_sizes:
            v_layers += [nn.Linear(last_size, size), nn.Tanh()]
            last_size = size
        v_layers += [nn.Linear(last_size, 1)]
        self.value_net = nn.Sequential(*v_layers)

        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-5)

    def forward(self, obs: torch.Tensor):
        mean = self.policy_mean(obs)
        logstd = self.policy_logstd.expand_as(mean)
        std = torch.exp(logstd)
        return Normal(mean, std)

    def get_action_and_value(self, obs: torch.Tensor, action=None):
        dist = self.forward(obs)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_net(obs).squeeze(-1)
        return action, logp, entropy, value

    def compute_gae(self, rewards, values, dones, last_value):
        T, N = rewards.shape
        adv = torch.zeros_like(rewards, device=self.device)
        lastgaelam = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            nextval = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nextval * nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + values
        return adv, returns

    def update(self, batch):
        obs, actions, old_logp, returns, advs = batch
        batch_size = obs.shape[0]
        mb_size = batch_size // self.minibatches

        for _ in range(self.epochs):
            idxs = np.random.permutation(batch_size)
            for start in range(0, batch_size, mb_size):
                mb_idx = idxs[start:start + mb_size]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advs = advs[mb_idx]
                mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                dist = self.forward(mb_obs)
                mb_logp = dist.log_prob(mb_actions).sum(-1)
                mb_entropy = dist.entropy().sum(-1)
                mb_value = self.value_net(mb_obs).squeeze(-1)

                ratio = (mb_logp - mb_old_logp).exp()
                pg_loss = -torch.min(
                    ratio * mb_advs,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advs,
                ).mean()

                v_unclipped = (mb_value - mb_returns) ** 2
                v_clipped = (mb_value + torch.clamp(mb_value - mb_returns, -self.clip_eps, self.clip_eps) - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()

                entropy_loss = -mb_entropy.mean()
                loss = pg_loss + self.value_coeff * v_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save_model(self, filepath):
        """Save the complete model (architecture + weights)"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'clip_eps': self.clip_eps,
            'gamma': self.gamma,
            'lam': self.lam,
            'epochs': self.epochs,
            'minibatches': self.minibatches,
            'entropy_coeff': self.entropy_coeff,
            'value_coeff': self.value_coeff,
            'max_grad_norm': self.max_grad_norm,
        }, filepath)
        print(f"✓ Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        """Load the complete model"""
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(
            obs_dim=checkpoint['obs_dim'],
            act_dim=checkpoint['act_dim'],
            clip_eps=checkpoint['clip_eps'],
            gamma=checkpoint['gamma'],
            lam=checkpoint['lam'],
            epochs=checkpoint['epochs'],
            minibatches=checkpoint['minibatches'],
            entropy_coeff=checkpoint['entropy_coeff'],
            value_coeff=checkpoint['value_coeff'],
            max_grad_norm=checkpoint['max_grad_norm'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def learn(self, envs, total_steps, steps_per_env, log_dir="./logs", log_interval=1000):
        num_envs = envs.num_envs
        obs_shape = envs.single_observation_space.shape
        device = self.device

        # buffers
        obs_buf = torch.zeros((steps_per_env, num_envs) + obs_shape, device=device)
        act_buf = torch.zeros((steps_per_env, num_envs, self.act_dim), device=device)
        logp_buf = torch.zeros((steps_per_env, num_envs), device=device)
        rew_buf = torch.zeros((steps_per_env, num_envs), device=device)
        done_buf = torch.zeros((steps_per_env, num_envs), device=device)
        val_buf = torch.zeros((steps_per_env, num_envs), device=device)

        # reset
        obs, _ = envs.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
        done = torch.zeros(num_envs, device=device)

        total_steps_done = 0
        batch_size = num_envs * steps_per_env
        os.makedirs(log_dir, exist_ok=True)
        monitor_file = os.path.join(log_dir, "monitor.csv")

        # write header
        with open(monitor_file, "w") as f:
            f.write("episode,r,l\n")

        episode_rewards = np.zeros(num_envs)
        episode_lengths = np.zeros(num_envs)
        all_rewards, all_lengths = [], []

        # Calculate total number of updates for progress bar
        total_updates = total_steps // batch_size
        
        # Progress bar setup
        pbar = tqdm(total=total_updates, desc="Training Progress", 
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        update_count = 0
        while total_steps_done < total_steps:
            # rollout
            for t in range(steps_per_env):
                obs_buf[t] = obs
                done_buf[t] = done

                with torch.no_grad():
                    action, logp, _, value = self.get_action_and_value(obs)
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value

                next_obs, rewards, terms, truncs, _ = envs.step(action.cpu().numpy())
                obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
                done = torch.tensor(np.logical_or(terms, truncs), device=device)
                rew_buf[t] = torch.tensor(rewards, device=device)

                # track per-env episodes
                episode_rewards += rewards
                episode_lengths += 1
                for i, d in enumerate(done.cpu().numpy()):
                    if d:
                        all_rewards.append(episode_rewards[i])
                        all_lengths.append(episode_lengths[i])
                        with open(monitor_file, "a") as f:
                            f.write(f"{len(all_rewards)},{episode_rewards[i]},{episode_lengths[i]}\n")
                        episode_rewards[i] = 0.0
                        episode_lengths[i] = 0

            with torch.no_grad():
                _, _, _, last_val = self.get_action_and_value(obs)
            advs, returns = self.compute_gae(rew_buf, val_buf, done_buf, last_val)

            # flatten
            b_obs = obs_buf.reshape(-1, *obs_shape)
            b_act = act_buf.reshape(-1, self.act_dim)
            b_logp = logp_buf.reshape(-1)
            b_returns = returns.reshape(-1)
            b_advs = advs.reshape(-1)

            decay_factor = update_count / total_updates  # goes from 0 → 1
            self.entropy_coeff = self.initial_entropy_coef + (self.final_entropy_coef - self.initial_entropy_coef) * decay_factor

            self.update((b_obs, b_act, b_logp, b_returns, b_advs))
            total_steps_done += batch_size
            update_count += 1

            # Update progress bar with additional info
            if len(all_rewards) >= 1:
                mean_reward = np.mean(all_rewards[-10:])
                pbar.set_postfix({
                    'Steps': f'{total_steps_done:,}',
                    'Episodes': len(all_rewards),
                    'Mean Reward': f'{mean_reward:.2f}'
                })
            
            pbar.update(1)

            # Detailed terminal log (less frequent)
            if total_steps_done % log_interval == 0 and len(all_rewards) >= 1:
                mean_r = np.mean(all_rewards[-10:])
                mean_l = np.mean(all_lengths[-10:])
                tqdm.write(f"Steps: {total_steps_done:,} | Mean Reward (last 10): {mean_r:.2f} | "
                          f"Mean Length (last 10): {mean_l:.1f} | Episodes: {len(all_rewards)}")

        pbar.close()
        
        # Save model with proper filename
        model_path = os.path.join(log_dir, "ppo_go2_model.pth")
        self.save_model(model_path)
        
        print(f"✓ Training completed!")
        print(f"✓ Total episodes: {len(all_rewards)}")
        if all_rewards:
            print(f"✓ Final mean reward: {np.mean(all_rewards[-10:]):.2f}")

        return monitor_file

# === Entry point ===
if __name__ == "__main__":
    # Hyperparameters
    TOTAL_TIMESTEPS = 1_00_00_000
    STEPS_PER_ENV = 2048
    NUM_ENVS = 4
    LOG_DIR = "./logs"

    # Create vectorized envs
    def make_env():
        return Go2Env(render_mode=None)

    envs = SyncVectorEnv([make_env for _ in range(NUM_ENVS)])

    # Instantiate agent
    sample_env = Go2Env(render_mode=None)
    obs_dim = sample_env.observation_space.shape[0]
    act_dim = sample_env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(64, 64),
        clip_eps=0.12,
        gamma=0.99,
        lam=0.95,
        lr=2e-4,
        epochs=10,
        minibatches=NUM_ENVS // 2,
        initial_entropy_coeff = 0.001,
        final_entropy_coeff = 0.00005,
        value_coeff=0.5,
        max_grad_norm=0.5,
        device=device,
    )

    print("Starting custom PPO training without Stable Baselines...")
    print(f"Device: {device}")
    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps with {NUM_ENVS} environments")
    print("-" * 60)
    
    monitor_csv = agent.learn(envs, TOTAL_TIMESTEPS, STEPS_PER_ENV, log_dir=LOG_DIR, log_interval=10000)

    # === Generate Training Graphs ===
    print("\nGenerating training graphs...")
    df = pd.read_csv(monitor_csv)
    fig, axes = plt.subplots(2, 2, figsize=(14,10))

    # Episode rewards
    axes[0,0].plot(df['r'], alpha=0.7)
    axes[0,0].set_title("Episode Rewards")
    axes[0,0].set_xlabel("Episode")
    axes[0,0].set_ylabel("Reward")
    axes[0,0].grid(True, alpha=0.3)

    # Moving average
    window=10
    axes[0,1].plot(df['r'], alpha=0.3, label='Raw')
    axes[0,1].plot(df['r'].rolling(window).mean(), color='red', label=f"{window}-ep MA")
    axes[0,1].set_title("Moving Average of Rewards")
    axes[0,1].set_xlabel("Episode")
    axes[0,1].set_ylabel("Reward")
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Episode lengths
    axes[1,0].plot(df['l'], alpha=0.7, color='green')
    axes[1,0].set_title("Episode Lengths")
    axes[1,0].set_xlabel("Episode")
    axes[1,0].set_ylabel("Length")
    axes[1,0].grid(True, alpha=0.3)

    # Reward histogram
    axes[1,1].hist(df['r'], bins=20, color='orange', edgecolor='black')
    axes[1,1].set_title("Reward Distribution")
    axes[1,1].set_xlabel("Reward")
    axes[1,1].set_ylabel("Frequency")
    axes[1,1].grid(True, alpha=0.3)

    # Add statistics
    stats_text = f"""Training Statistics:
    Total Episodes: {len(df)}
    Mean Reward: {df['r'].mean():.2f}
    Max Reward: {df['r'].max():.2f}
    Min Reward: {df['r'].min():.2f}
    Mean Episode Length: {df['l'].mean():.1f} steps"""
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    graph_path = os.path.join(LOG_DIR, "training_summary.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Graphs saved to {graph_path}")
    print(f"✓ Model saved to {LOG_DIR}/ppo_go2_model.pth")
    print(f"✓ Training logs saved to {LOG_DIR}/monitor.csv")
