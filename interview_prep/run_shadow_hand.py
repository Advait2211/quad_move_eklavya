import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("shadow_hand.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Shadow-like hand loaded. Close the viewer to exit.")
    step = 0
    while viewer.is_running():
        # Wiggle fingers for demo
        step += 1
        time.sleep(1)
        for i in range(model.nu):
            data.ctrl[i] = 0.5 * np.sin(0.01 * step + i)
        mujoco.mj_step(model, data)
        viewer.sync()
