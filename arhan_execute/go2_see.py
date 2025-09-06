import torch
import numpy as np
import gymnasium as gym
from torch.distributions import Normal
import time

# ========================
# CONFIG
# ========================
ENV_ID = "Ant-v5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# ENV CREATION
# ========================
def make_env():
    def _init():
        return gym.make(
            ENV_ID,
            xml_file="./unitree_go2/scene.xml",
            forward_reward_weight=0.32 / 0.32,   # keep base scale at 1
            ctrl_cost_weight=0.02 / 0.32,        # relative to forward term
            contact_cost_weight=0.03 / 0.32,     # absorb joint+smooth penalties
            healthy_reward=0.05 / 0.32,          # alive bonus relative scale
            main_body=1,
            healthy_z_range=(0.20, 0.70),        # match your termination bounds
            include_cfrc_ext_in_observation=True,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.01,
            frame_skip=2,
            max_episode_steps=1000,
            render_mode="human",
        )
    return _init

# ========================
# MODEL ARCHITECTURE
# ========================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(torch.nn.Module):
    def __init__(self, envs):
        super().__init__()
        # FIX: use observation_space and action_space directly
        obs_shape = envs.observation_space.shape
        action_shape = envs.action_space.shape
        
        # Critic network
        self.critic = torch.nn.Sequential(
            layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 1), std=1.0),
        )

        # Actor network
        self.actor_mean = torch.nn.Sequential(
            layer_init(torch.nn.Linear(np.array(obs_shape).prod(), 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, np.prod(action_shape)), std=0.01),
        )
        
        # Learnable log std
        self.actor_logstd = torch.nn.Parameter(torch.zeros(1, np.prod(action_shape)))

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
# MAIN VISUALIZATION SCRIPT
# ========================
def main():
    # Create dummy env for shapes
    dummy_env = make_env()()
    agent = Agent(dummy_env).to(DEVICE)

    # Ask user what type of model to load
    model_type = input("Enter model type to visualize ('checkpoint' or 'final'): ").strip().lower()
    model_path = input("Enter the path to the model file: ").strip()

    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    if model_type == "checkpoint":
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        agent.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    elif model_type == "final":
        # Load final weights only
        agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded final trained model.")
    else:
        print("Invalid input! Please enter 'checkpoint' or 'final'.")
        return

    agent.eval()  # Put model in evaluation mode

    # Start visualization
    env = make_env()()
    NUM_EVAL_EPISODES = 5

    for ep in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t)
                action = action.cpu().numpy()[0]
            obs, reward, term, trunc, _ = env.step(action)
            ep_return += reward
            done = bool(term or trunc)
        print(f"Episode {ep + 1}: Return = {ep_return:.2f}")

    env.close()
    dummy_env.close()

if __name__ == "__main__":
    main()
