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

        # --- Base state ---
        base_lin_vel = self.data.qvel[0:2]      # x, y linear velocity
        height = self.data.qpos[2]
        roll, pitch, yaw = self.data.qpos[4], self.data.qpos[5], self.data.qpos[3]
        joint_dev = np.abs(self.data.qpos[7:] - self.default_dof_pos)

        # --- Forward velocity & tracking reward ---
        target_vel = np.array([0.5, 0.0])
        lin_vel_error = np.sum((target_vel - base_lin_vel)**2)
        tracking_reward = 1.0 * base_lin_vel[0] + 2.0 * np.exp(-lin_vel_error / 0.25)

        # --- Height reward (Gaussian around 0.45m) ---
        height_bonus = np.exp(-((height - 0.45)**2) / 0.02)

        # --- Orientation penalties ---
        excess_roll = max(0.0, abs(roll) - 0.10)
        excess_pitch = max(0.0, abs(pitch) - 0.15)
        excess_yaw = max(0.0, abs(yaw) - 0.02)
        roll_penalty = -0.4 * excess_roll**2
        pitch_penalty = -0.4 * excess_pitch**2
        yaw_penalty = -0.3 * excess_yaw**2

        # --- Joint regularization ---
        joint_reg_penalty = -0.1 * np.sum(joint_dev**2)

        # --- Control cost ---
        ctrl_cost = -0.001 * np.sum(action**2)

        # --- Survival / alive bonus ---
        alive_bonus = 1.0
        survival_bonus = 0.1

        # --- Foot contacts & diagonal gait ---
        foot_geom_ids = {f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f)
                        for f in ["FL", "FR", "RL", "RR"]}
        foot_contact = {f: False for f in foot_geom_ids}
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for f, gid in foot_geom_ids.items():
                if c.geom1 == gid or c.geom2 == gid:
                    foot_contact[f] = True

        diag1 = foot_contact["FL"] and foot_contact["RR"]
        diag2 = foot_contact["FR"] and foot_contact["RL"]
        current_diag = None
        if diag1:
            current_diag = "diag1"
        elif diag2:
            current_diag = "diag2"

        gait_reward = 0.0
        if current_diag is not None:
            gait_reward = 0.2 * base_lin_vel[0] + 0.05 * self.same_diag_count
            if current_diag == self.last_diag:
                self.same_diag_count += 1
            else:
                self.same_diag_count = 1
            self.last_diag = current_diag
        else:
            self.same_diag_count = 0
            self.last_diag = None

        gait_reward += 0.5 if current_diag else -0.2

        # Additional gait penalty for number of grounded feet
        grounded_feet = sum(foot_contact.values())
        if grounded_feet == 2:
            gait_reward += 0.2
        elif grounded_feet < 2:
            gait_reward -= 0.5

        # Prevent stuck in same diagonal too long
        if self.same_diag_count > 10:
            gait_reward -= 0.05 * (self.same_diag_count - 10)

        # --- Forward progress reward ---
        forward_progress = self.data.qpos[0] - getattr(self, 'last_base_x', 0.0)
        self.last_base_x = self.data.qpos[0]

        # --- Total reward ---
        reward = (
            0.5 * tracking_reward +
            0.3 * height_bonus +
            0.2 * gait_reward +
            0.1 * survival_bonus +
            0.2 * forward_progress +
            alive_bonus +
            roll_penalty + pitch_penalty + yaw_penalty +
            joint_reg_penalty +
            ctrl_cost
        )

        # --- Termination ---
        terminated = True if height < 0.15 or height > 0.8 else False
        truncated = self.current_steps >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, {}


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
            """
            how the bot is currently moving. 
            this is the current reward function, and here is what the bot is doing: after startup, the bot pulls its front legs behind (both of them, and places them closer, side by side). by doing this the bot tilts forwards and goes a little bit forward. then using a little bit jumping of rear legs it tries to move forward. so it falls down either rolling on its face, or one of the front legs come off balance and then it falls to one side. another worth mentioning is that it spreads its rear legs, and its front legs it brings closer.
            """