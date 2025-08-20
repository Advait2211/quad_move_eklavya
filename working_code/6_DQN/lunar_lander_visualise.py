import torch
import gymnasium as gym
import numpy as np


class DQN(torch.nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

# --- Load environment with rendering ---
env = gym.make("LunarLander-v3", render_mode="human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# --- Load the trained model ---
policy_net = DQN(state_size, action_size)
policy_net.load_state_dict(torch.load("dqn_lunarlander_parallel.pth", map_location=torch.device("cpu")))
policy_net.eval()

# --- Run and visualise episodes ---
NUM_EPISODES = 5
for episode in range(NUM_EPISODES):
    state = env.reset(seed=episode)[0]
    done = False
    total_reward = 0

    while not done:
        # Choose the best action
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        # Step in the environment
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Episode {episode+1}: Total Reward = {total_reward:.2f}")

env.close()
