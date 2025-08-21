import gymnasium as gym
import torch
import torch.nn as nn
from torch.distributions import Normal

# --- CONFIG ---
number = 1100
MODEL_PATH = f"./ppo_go2_update_{number}.pth"   # your saved PPO model file
ENV_ID = "Ant-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ActorCritic (same as training) ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden = 256
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
        value = self.critic(x)
        mean = self.actor(x)
        std = torch.exp(self.log_std)
        return mean, std, value

# --- Env creation ---
def make_env():
    return gym.make(
        ENV_ID,
        xml_file="/Users/advaitdesai/Programming/eklavya/week7/go2_walk/unitree_go2/scene.xml",
        forward_reward_weight=2,
        ctrl_cost_weight=0.1,
        contact_cost_weight=0.01,
        healthy_reward=1,
        main_body=1,
        healthy_z_range=(0.3, 0.65),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.01,
        frame_skip=2,
        max_episode_steps=200,        
        render_mode="human",  # important for visualization
        # camera_name="lookat",
)

# --- Load environment ---
env = make_env()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# --- Load trained model ---
model = ActorCritic(state_dim, action_dim).to(DEVICE)

# Load checkpoint properly
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # fallback in case it's a direct state_dict
    model.load_state_dict(checkpoint)

model.eval()

# --- Run evaluation ---
obs, _ = env.reset()
done = False
total_reward = 0

while True:
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mean, std, _ = model(obs_t)
        dist = Normal(mean, std)
        action = dist.mean.cpu().numpy()[0]  # deterministic policy
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        print(f"Episode reward: {total_reward:.2f}")
        obs, _ = env.reset()
        total_reward = 0
