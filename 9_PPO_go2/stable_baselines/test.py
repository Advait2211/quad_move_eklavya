import time
from stable_baselines3 import PPO
from go2_env import Go2Env

# Use render_mode="human" to actually open the MuJoCo viewer
env = Go2Env(render_mode="human")

# Load the trained model (use .zip file, not folder)
model = PPO.load("ppo_go2.zip", env=env)

num_episodes = 5
for ep in range(num_episodes):
    obs, _ = env.reset()
    done, truncated = False, False
    episode_reward = 0
    steps = 0
    while not (done or truncated):
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        # Important: this calls mujoco.viewer.sync()
        env.render()

        # Slow it down a bit so you can actually watch
        time.sleep(1.0 / 60.0)

    print(f"Episode {ep+1} reward: {episode_reward:.2f} steps: {steps}")

env.close()
