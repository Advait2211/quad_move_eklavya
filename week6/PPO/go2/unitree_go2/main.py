import gymnasium as gym
import numpy as np
import time

# Basic loading (uncomment to use)
# env = gym.make('Ant-v5', xml_file='./mujoco_menagerie/unitree_go1/scene.xml')

# Although this is enough to load the model, we will need to tweak some environment parameters
# to get the desired behavior for our environment, so we will also explicitly set the simulation,
# termination, reward and observation arguments, which we will tweak in the next step.

env = gym.make(
    "Ant-v5",
    xml_file="./scene.xml",
    forward_reward_weight=0,
    ctrl_cost_weight=0,
    contact_cost_weight=0,
    healthy_reward=0,
    main_body=1,
    healthy_z_range=(0, np.inf),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=100,
    render_mode="human"
)

# Reset the environment
obs, info = env.reset()

# Run a few steps with random actions
for _ in range(500):
    action = env.action_space.sample()  # or your policy
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
        

env.close()