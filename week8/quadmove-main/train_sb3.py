import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import your custom environment
from go2_env import Go2Env

# Custom callback for terminal logging
class TerminalLoggerCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super(TerminalLoggerCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Get reward and done info
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
        # Log progress to terminal
        if self.n_calls % self.check_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
                mean_length = np.mean(self.episode_lengths[-10:])
                print(f"Step: {self.num_timesteps} | "
                      f"Mean Reward (last 10): {mean_reward:.2f} | "
                      f"Mean Length: {mean_length:.1f} | "
                      f"Total Episodes: {len(self.episode_rewards)}")
        
        return True

# Create and wrap your environment
env = Go2Env(render_mode=None)
env = Monitor(env, filename="./logs/monitor")  # Still save for final graphs

# Check if the environment follows the Gym interface
check_env(env)

# Vectorize environment
env = DummyVecEnv([lambda: env])

# Create callback
terminal_callback = TerminalLoggerCallback(check_freq=1000)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    tensorboard_log="./ppo_go2_tensorboard/"
)

print("Starting training...")
print("Progress will be logged to terminal every 1000 steps")
print("-" * 60)

# Train the model
model.learn(
    total_timesteps=1500000,
    callback=terminal_callback,
    progress_bar=True
)

# Save the model
model.save("ppo_go2")
print("\nTraining completed! Model saved as 'ppo_go2'")

# ===== GENERATE FINAL GRAPHS =====
print("\nGenerating training graphs...")

def generate_training_graphs():
    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)
    
    try:
        # Read monitor data
        monitor_data = pd.read_csv("./logs/monitor.monitor.csv", skiprows=1)
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode Rewards
        ax1.plot(monitor_data['r'], alpha=0.7, linewidth=1)
        ax1.set_title('Episode Rewards Over Time')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Moving Average of Rewards (window=10)
        window = 10
        moving_avg = monitor_data['r'].rolling(window=window).mean()
        ax2.plot(monitor_data['r'], alpha=0.3, label='Raw', linewidth=1)
        ax2.plot(moving_avg, label=f'{window}-episode MA', color='red', linewidth=2)
        ax2.set_title('Moving Average of Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Episode Lengths
        ax3.plot(monitor_data['l'], alpha=0.7, color='green', linewidth=1)
        ax3.set_title('Episode Lengths')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Steps')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Reward Distribution Histogram
        ax4.hist(monitor_data['r'], bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        stats_text = f"""Statistics:
        Total Episodes: {len(monitor_data)}
        Mean Reward: {monitor_data['r'].mean():.2f}
        Max Reward: {monitor_data['r'].max():.2f}
        Min Reward: {monitor_data['r'].min():.2f}
        Mean Length: {monitor_data['l'].mean():.1f} steps"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        plt.savefig('./logs/training_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Graphs saved to './logs/training_summary.png'")
        print(f"✓ Total episodes trained: {len(monitor_data)}")
        print(f"✓ Average reward: {monitor_data['r'].mean():.2f}")
        print(f"✓ Best episode reward: {monitor_data['r'].max():.2f}")
        
    except FileNotFoundError:
        print("⚠️  Monitor data not found. No graphs generated.")
    except Exception as e:
        print(f"⚠️  Error generating graphs: {e}")

# Generate the graphs
generate_training_graphs()

