import gymnasium as gym


def main():
    """Run the final Go1 environment setup."""
    # Note: The original tutorial includes an image showing the Go1 robot in the environment.
    # The image is available at: https://github.com/Kallinteris-Andreas/Gymnasium-kalli/assets/30759571/bf1797a3-264d-47de-b14c-e3c16072f695

    env = gym.make(
        "Ant-v5",
        xml_file="./scene.xml",
        forward_reward_weight=1,
        ctrl_cost_weight=0.05,
        contact_cost_weight=5e-4,
        healthy_reward=1,
        main_body=1,
        healthy_z_range=(0.195, 0.75),
        include_cfrc_ext_in_observation=True,
        exclude_current_positions_from_observation=False,
        reset_noise_scale=0.1,
        frame_skip=25,
        max_episode_steps=1000,
        render_mode="human",  # Change to "human" to visualize
    )

    # Example of running the environment for a few steps
    obs, info = env.reset()

    for _ in range(100):
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    print("Environment tested successfully!")

main()