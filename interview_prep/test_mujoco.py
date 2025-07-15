import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <geom type="sphere" size="0.05" pos="0 0 0.05" rgba="1 0 0 1"/>
  </worldbody>
</mujoco>
""")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Close the viewer window to end.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1000)
