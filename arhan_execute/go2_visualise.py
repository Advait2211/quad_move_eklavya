import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import os
import glob
import time

# ========================
# MODEL DEFINITION (Same as training)
# ========================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        
        # Critic network
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        
        # Actor network
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        
        # Learnable log std
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def load_latest_checkpoint(checkpoint_dir="./checkpoints", device="cpu"):
    """Load the latest checkpoint from the directory"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ppo_go2_iter_*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return None, None
    
    # Sort by iteration number and get the latest
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    latest_checkpoint = checkpoint_files[-1]
    
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    iteration = checkpoint.get("iteration", "unknown")
    
    print(f"Loading checkpoint from iteration {iteration}: {latest_checkpoint}")
    return checkpoint, iteration

def load_specific_checkpoint(checkpoint_path, device="cpu"):
    """Load a specific checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    iteration = checkpoint.get("iteration", "unknown")
    
    print(f"Loading checkpoint from iteration {iteration}: {checkpoint_path}")
    return checkpoint, iteration

def create_render_env():
    """Create environment with rendering enabled"""
    env = gym.make(
        'Ant-v5',
        xml_file="./unitree_go2/scene(friction).xml",
        forward_reward_weight=4.0,       # Increase forward reward strength to push forward motion
        ctrl_cost_weight=0.05,            # Decrease control cost to allow more flexible actuation
        contact_cost_weight=0.005,        # Reduce penalty on contact forces to avoid discouraging foot contact
        healthy_reward=1.5,               # Increase healthy reward to more strongly encourage upright posture
        main_body=1,
        healthy_z_range=(0.35, 0.52),    # Narrow healthy height range to penalize collapsing or overextension
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.01,
        frame_skip=2,
        max_episode_steps=1000,
        render_mode='human',
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    return env

def visualize_robot(checkpoint_path=None, num_episodes=5, max_steps=2000):
    """Visualize the trained robot"""
    
    device = torch.device("cpu")  # Use CPU for visualization
    
    # Load checkpoint
    if checkpoint_path:
        checkpoint, iteration = load_specific_checkpoint(checkpoint_path, device)
    else:
        checkpoint, iteration = load_latest_checkpoint(device=device)
    
    if checkpoint is None:
        return
    
    # Create environment
    print("Creating environment...")
    env = create_render_env()
    
    # Initialize agent
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    agent = Agent(obs_shape, action_shape).to(device)
    
    # Load model weights
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    
    print(f"\n🚀 Starting visualization of Go2 robot (Iteration {iteration})")
    print(f"Running {num_episodes} episodes...")
    print("Press Ctrl+C to stop early\n")
    
    episode_returns = []
    
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            step_count = 0
            
            print(f"Episode {episode + 1}/{num_episodes} - ", end="", flush=True)
            
            while step_count < max_steps:
                # Render the environment
                env.render()
                
                # Get action from trained policy
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                    action = action.cpu().numpy()[0]
                
                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                step_count += 1
                
                # Add small delay to make visualization smoother
                time.sleep(0.01)
                
                if terminated or truncated:
                    break
            
            episode_returns.append(episode_return)
            print(f"Return: {episode_return:.2f}, Steps: {step_count}")
            
            # Small pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    
    finally:
        env.close()
    
    if episode_returns:
        print(f"\nVisualization Summary:")
        print(f"Mean Return: {np.mean(episode_returns):.2f}")
        print(f"Std Return: {np.std(episode_returns):.2f}")
        print(f"Episodes completed: {len(episode_returns)}")

def list_available_checkpoints(checkpoint_dir="./checkpoints"):
    """List all available checkpoints"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "ppo_go2_iter_*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return []
    
    # Sort by iteration number
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    print("Available checkpoints:")
    for i, checkpoint_path in enumerate(checkpoint_files):
        iteration = checkpoint_path.split("_")[-1].split(".")[0]
        print(f"{i+1}. Iteration {iteration}: {checkpoint_path}")
    
    return checkpoint_files

# ========================
# MAIN SCRIPT
# ========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize trained Go2 robot")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to specific checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=2000, 
                       help="Maximum steps per episode")
    parser.add_argument("--list", action="store_true", 
                       help="List available checkpoints")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints()
    else:
        visualize_robot(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            max_steps=args.max_steps
        )