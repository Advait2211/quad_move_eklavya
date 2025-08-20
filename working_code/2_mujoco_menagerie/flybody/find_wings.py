import mujoco
from mujoco import mjx
import gymnasium as gym

env = gym.make("Ant-v5", xml_file="./scene.xml")
model = env.unwrapped.model  # Access the MuJoCo model

# Print actuator names and their indices
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"{i:02d}: {name}")
