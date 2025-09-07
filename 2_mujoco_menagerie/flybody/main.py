import gymnasium as gym
import numpy as np
import time

env = gym.make(
    "Ant-v5",
    xml_file="./scene.xml",
    render_mode="human",
    frame_skip=5,
    max_episode_steps=1000,
)

obs, info = env.reset()

t = 0
while t < 200:
    # Sinusoidal action pattern
    action = np.zeros(env.action_space.shape)
    
    # Let's assume first 2 actuators control "wings"
    action[0] = 0.5 * np.sin(t * 0.1)
    action[1] = -0.5 * np.sin(t * 0.1)

    obs, reward, terminated, truncated, info = env.step(action)
    # time.sleep(0.05)
    t += 1

    if terminated or truncated:
        obs, info = env.reset()
        t = 0

env.close()
