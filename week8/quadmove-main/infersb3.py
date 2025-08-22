import gymnasium as gym
from stable_baselines3 import PPO
import time
from go2_env import Go2Env

def test_trained_model():
    print("Loading trained model...")
    
    # Load the trained model
    model = PPO.load("ppo_go2")
    
    print("Creating test environment with rendering...")
    
    # Create environment with rendering (NO vectorization for testing)
    env = Go2Env(render_mode="human")
    
    print("Starting inference...")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    obs, _ = env.reset()
    total_reward = 0
    episode_count = 0
    
    try:
        while True:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Render (this will show the simulation)
            env.render()
            
            # Small delay to make it viewable
            time.sleep(0.01)
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count}: Total reward = {total_reward:.2f}")
                total_reward = 0
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping inference...")
    
    finally:
        # Close the viewer if it exists
        if hasattr(env, 'viewer') and env.viewer is not None:
            env.viewer.close()
        print("Environment closed.")

if __name__ == "__main__":
    test_trained_model()