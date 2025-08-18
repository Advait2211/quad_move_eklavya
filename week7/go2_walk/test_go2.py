import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
import time
import imageio

# ========================
# CONFIG
# ========================
ENV_ID = "Ant-v5"
MODEL_PATH = "ppo_go2_update_75.pth"
RENDER = False
RECORD = False
RENDER_MODE = "rgb_array" if RECORD else "human"
EPISODE_LENGTH = 1000  # Max steps per episode
VIDEO_PATH = "ppo_go2_eval.mp4"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# ENV SETUP
# ========================
def make_env():
    return gym.make(
        ENV_ID,
        xml_file="./unitree_go2/scene.xml",
        forward_reward_weight=25,
        ctrl_cost_weight=0.10,
        contact_cost_weight=-0.001,
        healthy_reward=5,
        main_body=1,
        healthy_z_range=(0.25, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.05,
        frame_skip=2,
        max_episode_steps=EPISODE_LENGTH,
        render_mode=RENDER_MODE,
    )

# ========================
# MODEL DEFINITION
# ========================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 1024
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        value = self.critic(x)
        return mean, std, value

# ========================
# EVALUATION LOOP
# ========================
def evaluate():
    EPISODES = 10
    for _ in range(EPISODES):
        env = make_env()
        obs, _ = env.reset()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        model = ActorCritic(state_dim, action_dim).to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        frames = []
        total_reward = 0.0
    
        for step in range(EPISODE_LENGTH):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                mean, _, _ = model(obs_t)
                action = mean.squeeze().cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if RECORD:
                frame = env.render()
                frames.append(frame)
            elif RENDER:
                env.render()
                time.sleep(1 / 30)  # 30 FPS

            if terminated or truncated:
                print(f"Episode ended after {step + 1} steps with reward: {total_reward:.2f}")
                break

        env.close()

    if RECORD:
        print(f"Saving video to {VIDEO_PATH} ...")
        imageio.mimsave(VIDEO_PATH, frames, fps=30)
        print("Done.")

# ========================
# MAIN
# ========================
if __name__ == "__main__":
    evaluate()
