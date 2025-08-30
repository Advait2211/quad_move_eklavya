import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces
import sys

class Go2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="./unitree_go2/scene(friction).xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.last_diag = None
        self.same_diag_count = 0

        print(self.data.qpos)
        # sys.exit()

        # Action scaling from XML
        self.n_joints = self.model.nu
        self.action_low = self.model.actuator_ctrlrange[:, 0]
        self.action_high = self.model.actuator_ctrlrange[:, 1]

        # Observation space: joint pos/vel, base quat, lin vel, ang vel
        obs_dim = (
            self.model.nq - 7 +  # exclude root pos+quat
            self.model.nv - 6 +  # exclude root lin+ang vel
            4 +                  # base quat
            3 +                  # base lin vel
            3                    # base ang vel
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_joints,), dtype=np.float32)
        
        # Episode tracking
        self.max_steps = 1000
        self.current_steps = 0

    def _get_obs(self):
        qpos = self.data.qpos[7:].ravel()
        qvel = self.data.qvel[6:].ravel()
        base_quat = self.data.qpos[3:7]
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        return np.concatenate([qpos, qvel, base_quat, base_linvel, base_angvel]).astype(np.float32)

    def step(self, action):
        # --- Apply action ---
        ctrl = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.current_steps += 1

        # --- Get base and joint state ---
        pos = self.data.qpos
        vel = self.data.qvel
        base_pos = pos[0:3]
        base_quat = pos[3:7]
        base_lin_vel = vel[0:3]
        base_ang_vel = vel[3:6]
        joint_vel = vel[6:]

        # --- Compute roll, pitch from quaternion ---
        from scipy.spatial.transform import Rotation as R
        # MuJoCo quat order is [w, x, y, z]
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        roll, pitch, _ = r.as_euler('xyz', degrees=False)

        # --- Reward components ---

        # 1) Forward velocity tracking (target 0.5 m/s)
        forward_vel = base_lin_vel[0]
        vel_error = (forward_vel - 0.5)**2
        r_vel = np.exp(-vel_error / 0.25)

        # 2) Height maintenance (target height 0.34 m)
        height_error = (base_pos[2] - 0.34)**2
        r_height = np.exp(-height_error / 0.02)

        # 3) Posture stability (minimize roll & pitch)
        r_posture = np.exp(-(roll**2 + pitch**2) / 0.05)

        # 4) Action smoothness
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(action)
        action_diff = action - self.prev_action
        r_smooth = np.exp(-np.sum(action_diff**2) / 0.1)
        self.prev_action = action.copy()

        # 5) Control cost (small penalty on large commands)
        r_ctrl = np.exp(-0.01 * np.sum(action**2))

        # 6) Alive bonus
        r_alive = 1.0

        # --- Combine with simple weights ---
        reward = (
            0.4 * r_vel +
            0.2 * r_height +
            0.2 * r_posture +
            0.1 * r_smooth +
            0.05 * r_ctrl +
            0.05 * r_alive
        )

        # --- Termination checks ---
        terminated = True if (
            base_pos[2] < 0.15 or base_pos[2] > 0.8 or
            abs(roll) > 1.0 or abs(pitch) > 1.0
        ) else False
        truncated = self.current_steps >= self.max_steps

        obs = self._get_obs()
        info = {
            'r_vel': r_vel,
            'r_height': r_height,
            'r_posture': r_posture
        }
        return obs, reward, terminated, truncated, info






    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_steps = 0

        # Add small random initial noise
        self.data.qpos[:] += np.random.uniform(-0.01, 0.01, size=self.model.nq)
        self.data.qvel[:] += np.random.uniform(-0.01, 0.01, size=self.model.nv)

        if not hasattr(self, "default_dof_pos") or self.default_dof_pos is None:
        # skip the floating base (first 7: 3 pos + 4 quat), take only joint angles
            self.default_dof_pos = np.copy(self.data.qpos[7:])

        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "human":
            # Use the older MuJoCo rendering approach
            try:
                # Try the modern viewer first
                import mujoco.viewer
                if not hasattr(self, 'viewer') or self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                else:
                    self.viewer.sync()
            except (AttributeError, ImportError):
                # Fallback: mujoco_py not available, raise error
                raise ImportError(
                    "Rendering failed: mujoco.viewer not available and mujoco_py could not be imported. "
                    "Please install mujoco or ensure your environment supports rendering."
                )