import mujoco
import mujoco.viewer
import numpy as np

# ✅ Load your model
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# ✅ Viewer launch on main thread
with mujoco.viewer.launch(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30
    viewer.cam.distance = 5.0

    t = 0
    while viewer.is_running():
        # ✅ Simple wing flapping logic
        flap = 1.5 * np.sin(t * 0.1)
        data.ctrl[:] = 0
        if model.nu >= 20:  # assuming actuator 16 and 19 exist
            data.ctrl[16] = 3.0 * np.sin(t * 0.15)
            data.ctrl[19] = 3.0 * np.sin(t * 0.15 + np.pi)

        mujoco.mj_step(model, data)
        viewer.sync()

        if t % 100 == 0:
            print(f"t={t} | z={data.qpos[2]:.2f}")
        t += 1
