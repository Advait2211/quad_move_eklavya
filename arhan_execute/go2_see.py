import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import os
import glob
import time
from gymnasium import spaces
import mujoco
from scipy.spatial.transform import Rotation as R
import argparse
import glfw
from mujoco import MjvCamera, MjvOption, MjvScene, mjv_updateScene, mjtCatBit, MjvPerturb


# ========================
# CUSTOM ENVIRONMENT WITH RENDERING
# ========================
class CustomAntEnv(gym.Env):
    def __init__(self, xml_file, max_steps=1000, render_mode=None, **kwargs):
        super().__init__()
        # load MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data  = mujoco.MjData(self.model)

        print(type(self.data))

        # spaces
        obs_dim = self.model.nq + self.model.nv + self.model.nbody * 6
        act_dim = self.model.nu
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0,   (act_dim,), np.float32)

        self.max_steps = max_steps
        self.cur_step  = 0
        self.prev_act  = None
        
        # Renderer and GLFW window (not initialized yet)
        self._renderer = mujoco.Renderer(self.model)
        self._window = None

        self._camera = MjvCamera()
        self._option = MjvOption()
        self._scene = MjvScene(self.model, maxgeom=10000)  # Allocate scene
        self._renderer = mujoco.Renderer(self.model)
        self._window = None
        self._perturb = MjvPerturb()

        mujoco.mjv_defaultCamera(self._camera)
        mujoco.mjv_defaultOption(self._option)

    def render(self, mode="human"):
        if self._window is None:
            if not glfw.init():
                raise RuntimeError("GLFW init failed")
            self._window = glfw.create_window(640, 480, "MuJoCo Simulation", None, None)
            if not self._window:
                glfw.terminate()
                raise RuntimeError("GLFW window creation failed")
            glfw.make_context_current(self._window)

        if glfw.window_should_close(self._window):
            self.close()
            return

        # Update scene with the current simulation data
        mujoco.mjv_updateScene(
            m=self.model,
            d=self.data,
            opt=self._option,
            pert=self._perturb,
            cam=self._camera,
            catmask=mujoco.mjtCatBit.mjCAT_ALL,
            scn=self._scene
        )

        self._renderer.update_scene(self._scene)
        self._renderer.render()

        glfw.swap_buffers(self._window)
        glfw.poll_events()

        time.sleep(1 / 60)

    def close(self):
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
            glfw.terminate()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.cur_step = 0
        self.prev_act = None
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
            self.data.cfrc_ext.flatten()
        ])

    def _compute_gait_action(self):
        return np.zeros(self.action_space.shape)

    def step(self, action):
        self.cur_step += 1

        gait  = self._compute_gait_action()
        mixed = 0.5 * gait + 0.5 * action

        self.data.ctrl[:] = np.clip(mixed, -1, 1)
        mujoco.mj_step(self.model, self.data)

        pos = self.data.qpos
        vel = self.data.qvel
        base_pos     = pos[0:3]
        base_quat    = pos[3:7]
        base_lin_vel = vel[0:3]
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        roll, pitch, _ = r.as_euler('xyz', degrees=False)

        # Custom reward calculation
        forward_vel = base_lin_vel[0]
        r_vel    = np.exp(-((forward_vel - 0.8)**2) / 0.5)
        r_height = np.exp(-((base_pos[2] - 0.34)**2) / 0.05)
        pitch_pen = 10.0 * pitch**2
        r_posture = np.exp(-(roll**2/0.02 + pitch_pen))
        r_joint   = np.exp(-0.05 * np.sum(pos[7:]**2))
        if self.prev_act is None:
            self.prev_act = mixed.copy()
        r_smooth = np.exp(-np.sum((mixed - self.prev_act)**2) / 0.2)
        self.prev_act = mixed.copy()
        r_ctrl    = np.exp(-0.005 * np.sum(mixed**2))
        r_alive   = 1.0
        r_lateral = np.exp(-0.5 * (base_lin_vel[1]**2))
        step_len  = abs(forward_vel) * (1/60)
        r_step    = np.tanh(step_len / 0.02)
        rear_hips = pos[7:][[4,6]]
        r_spread  = np.clip(np.mean(np.exp(-20.0*(rear_hips-0.05)**2)),1e-3,1.0)

        reward = (
            0.32*r_vel + 0.32*r_posture + 0.10*r_height +
            0.02*r_joint + 0.02*r_ctrl + 0.01*r_smooth +
            0.02*r_lateral + 0.05*r_alive + 0.12*r_spread +
            0.02*r_step
        )

        done = (
            base_pos[2] < 0.20 or base_pos[2] > 0.7 or
            abs(roll) > 1.0 or abs(pitch) > 0.6 or
            abs(base_lin_vel[1]) > 2.5 or
            self.cur_step >= self.max_steps
        )

        return self._get_obs(), reward, done, done, {}


# ========================
# MODEL DEFINITION 
# ========================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_shape, action_shape, model = None, data=None):
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

        self._renderer = mujoco.Renderer(self.model)
        self._window = None

        self.model = model
        self.data = data

        if self.model is not None:
            self._renderer = mujoco.Renderer(self.model)
        else:
            self._renderer = None
        self._window = None

    def render(self, mode="human"):
        # Create or reuse a GLFW window
        if self._window is None:
            self._window = glfw.create_window(640, 480, "MuJoCo CustomAntEnv", None, None)
            if not self._window:
                glfw.terminate()
                raise RuntimeError("Failed to create GLFW window")
            glfw.make_context_current(self._window)

        if glfw.window_should_close(self._window):
            self.close()
            return

        # Update MuJoCo scene & render
        self._renderer.update_scene(self.data)
        self._renderer.render()

        glfw.swap_buffers(self._window)
        glfw.poll_events()

    def close(self):
        if self._window:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

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


def visualize_robot(checkpoint_path=None, num_episodes=5, max_steps=2000, xml_path="./unitree_go2/scene(friction).xml"):
    """Visualize the trained robot using MuJoCo's passive viewer"""
    
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
    env = CustomAntEnv(xml_file=xml_path, max_steps=max_steps, render_mode="human")

    # Get environment dimensions
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    
    # Create and load agent
    agent = Agent(obs_shape, action_shape, model=env.model, data=env.data).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    
    print(f"\n🚀 Starting visualization of Go2 robot (Iteration {iteration})")
    print(f"Running {num_episodes} episodes with max {max_steps} steps each")
    print("MuJoCo viewer will open. Close the viewer window to stop.\n")
    
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
                
                if terminated or truncated:
                    break
            
            episode_returns.append(episode_return)
            print(f"Return: {episode_return:.2f}, Steps: {step_count}")
            
            # Small pause between episodes
            time.sleep(1.0)
    
    except KeyboardInterrupt:
        print("\nVisualization stopped by user")
    except Exception as e:
        print(f"\nError during visualization: {e}")
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


def test_environment_only(xml_path="./unitree_go2/scene(friction).xml", max_steps=1000):
    """Test the environment with random actions to verify rendering works"""
    print("Testing environment with random actions...")
    
    env = CustomAntEnv(xml_file=xml_path, max_steps=max_steps, render_mode="human")
    obs, _ = env.reset()
    
    try:
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Render
            env.render()
            
            print(f"Step {step}: Reward = {reward:.3f}")
            
            if terminated or truncated:
                print(f"Episode finished at step {step}")
                break
                
        print("Environment test completed successfully!")
        
    except KeyboardInterrupt:
        print("Test stopped by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        env.close()


# ========================
# MAIN SCRIPT
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained Go2 robot")
    parser.add_argument("--checkpoint", type=str, default=None, 
                       help="Path to specific checkpoint file")
    parser.add_argument("--episodes", type=int, default=5, 
                       help="Number of episodes to run")
    parser.add_argument("--max_steps", type=int, default=2000, 
                       help="Maximum steps per episode")
    parser.add_argument("--xml_path", type=str, default="./unitree_go2/scene(friction).xml",
                       help="Path to MuJoCo XML file")
    parser.add_argument("--list", action="store_true", 
                       help="List available checkpoints")
    parser.add_argument("--test", action="store_true",
                       help="Test environment with random actions")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints()
    elif args.test:
        test_environment_only(xml_path=args.xml_path)
    else:
        visualize_robot(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            xml_path=args.xml_path
        )