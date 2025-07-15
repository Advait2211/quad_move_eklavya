import brax
import functools
import jax
import threading
import time
import numpy as np
from jax import numpy as jp
from brax import envs
from brax.io import model
from brax.training.agents.ppo import train as ppo
import mujoco
import mujoco.viewer


env_name = 'ant'  
backend = 'positional'
env = envs.get_environment(env_name=env_name, backend=backend)

def progress(num_steps, metrics):
    print(f'Steps: {num_steps}, Reward: {metrics["eval/episode_reward"]:.2f}')

# Train the policy
train_fn = functools.partial(
    ppo.train,
    num_timesteps=5_000_000,
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

make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
model.save_params('ant_ppo_policy', params)
params = model.load_params('ant_ppo_policy')
inference_fn = make_inference_fn(params)
jit_inference_fn = jax.jit(inference_fn)

# Load MuJoCo model
xml_path = "ant.xml"
with open(xml_path, 'r') as f:
    xml_string = f.read()
model_mj = mujoco.MjModel.from_xml_string(xml_string)
data_mj = mujoco.MjData(model_mj)

def run_viewer():
    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        torso_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 7.0
        viewer.cam.azimuth = 0.0
        viewer.cam.elevation = -20.0
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


viewer_thread = threading.Thread(target=run_viewer, daemon=True)
viewer_thread.start()

# Run policy visualization
rng = jax.random.PRNGKey(0)
state = jax.jit(env.reset)(rng)
n_steps = 500
for _ in range(n_steps):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    data_mj.ctrl[:] = np.array(ctrl)  
    mujoco.mj_step(model_mj, data_mj)
    time.sleep(0.01)
