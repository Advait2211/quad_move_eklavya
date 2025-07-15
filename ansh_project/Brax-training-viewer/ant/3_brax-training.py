import brax
import functools
import jax
import time
import sys
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo

# Initialize the environment
env_name = 'ant'  
backend = 'generalized'  
env = envs.get_environment(env_name=env_name, backend=backend)

# Define PPO training function
num_timesteps = 1_000
train_fn = functools.partial(
    ppo.train,
    num_timesteps=num_timesteps,
    num_evals=10,
    reward_scaling=10,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=5,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=4096,
    batch_size=2048,
    seed=1
)

start_time = time.time()
logging_interval = 10  # Log every 2 minutes
last_log_time = start_time

make_inference_fn, params, training_metrics = train_fn(environment=env)
print("Final Reward:", training_metrics['eval/episode_reward'])


model.save_params('ant_ppo_policy', params)


for step in range(num_timesteps):
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    if current_time - last_log_time >= logging_interval:
        last_log_time = current_time
        progress = (step + 1) / num_timesteps
        estimated_total_time = elapsed_time / progress if progress > 0 else float('inf')
        estimated_time_remaining = estimated_total_time - elapsed_time
        
        print(f"Step: {step+1}/{num_timesteps}, Elapsed: {elapsed_time:.2f}s, Remaining: {estimated_time_remaining:.2f}s")
        print(f"Recent Reward Trend: {training_metrics['eval/episode_reward'] if 'eval/episode_reward' in training_metrics else 'N/A'}")
        sys.stdout.flush()

print("Training completed. Model saved as 'ant_ppo_policy'.")