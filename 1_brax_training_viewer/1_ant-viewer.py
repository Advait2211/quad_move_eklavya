import mujoco
import mujoco.viewer
import numpy as np
import time


model = mujoco.MjModel.from_xml_path("ant.xml")
data = mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    action_update_interval = 10  
    step_count = 0
    random_action = np.random.uniform(-5, 5, size=model.nu)

    while viewer.is_running():
        mujoco.mj_step(model, data)

        
        if step_count % action_update_interval == 0:
            random_action = np.random.uniform(-5, 5, size=model.nu)

        data.ctrl[:] = random_action
        step_count += 1

        viewer.sync()
        time.sleep(0.01) 

