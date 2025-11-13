import torch
import time
from go2_env import Go2Env
from train_go2 import PPOAgent

def visualize_model(model_path, device=None, render_mode="human", max_episodes=5, sleep_time=0.01):
    """
    Load trained PPOAgent model and run episodes in Go2Env with rendering.
    
    Args:
        model_path (str): Path to the saved model checkpoint (.pth).
        device (torch.device or None): Device to load the model on.
        render_mode (str): Rendering mode for environment ("human" for window, "rgb_array" for frames).
        max_episodes (int): Number of episodes to visualize.
        sleep_time (float): Delay between steps (seconds).
    """
    device = device or torch.device("cpu")

    # Load the model
    agent = PPOAgent.load_model(model_path, device=device)
    agent.eval()

    # Create environment with rendering enabled
    env = Go2Env(render_mode=render_mode)
    obs, _ = env.reset()

    for episode in range(max_episodes):
        done = False
        total_reward = 0.0
        obs = env.reset()[0]

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
            action = action.cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Render the environment
            frame = env.render()
            if render_mode == "rgb_array":
                # If rendering returns an image, you can display it using cv2 or matplotlib
                # Example (requires cv2):
                # import cv2
                # cv2.imshow("Go2 PPO Visualization", frame)
                # cv2.waitKey(1)
                pass
            else:
                # For "human" mode, env.render() typically creates an interactive window
                pass

            if terminated or truncated:
                done = True

            time.sleep(sleep_time)  # Slow down visualization for viewing

        print(f"Episode {episode + 1} finished with total reward: {total_reward:.2f}")

    env.close()

# Example usage:
if __name__ == "__main__":
    model_file = "./ppo_go2_model.pth"  # Adjust path as needed
    visualize_model(model_file, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), max_episodes=3)
