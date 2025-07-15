import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("so100_arm.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        for i in range(model.nu):
            data.ctrl[i] = 0.5 * np.sin(0.01 * step + i)
        mujoco.mj_step(model, data)
        viewer.sync()
        step += 1
        time.sleep(0.2)
