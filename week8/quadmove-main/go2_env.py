import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces

class Go2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="./unitree_go2/scene.xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.last_diag = None
        self.same_diag_count = 0

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
        # Rescale action from [-1, 1] to actuator range
        ctrl = self.action_low + (action + 1) * 0.5 * (self.action_high - self.action_low)
        self.data.ctrl[:] = ctrl

        mujoco.mj_step(self.model, self.data)
        self.current_steps += 1
        
        pitch = self.data.qpos[4]
        roll = self.data.qpos[3]
        
        # --- Reward shaping ---
        forward_vel = self.data.qvel[0]
        #forward_reward = forward_vel * 1 # Reward forward motion
        target_vel = np.array([0.5, 0.0])  # e.g. walk forward 0.5 m/s
        self.commands = target_vel

        # Actual base velocity
        base_lin_vel = self.data.qvel[0:2]  # x, y linear velocity

        # Error term
        lin_vel_error = np.sum((self.commands - base_lin_vel)**2)

        # Exponential tracking reward
        tracking_sigma = 0.225
        tracking_reward = np.exp(-lin_vel_error / tracking_sigma)

        height = self.data.qpos[2]
        height_bonus = 1.0 - abs(height - 0.45) * 2.0  # Less punitive height reward
        #height_bonus = 1.0 - ((height - 0.45) ** 2) * 5.0
        #height_bonus = 1.0 -((height - 0.45) ** 2) * 10.0

        joint_dev = np.abs(self.data.qpos[7:] - self.default_dof_pos)
        joint_reg_penalty = -0.15 * np.sum(joint_dev)

           # Reduced tilt penalty
        roll  = self.data.qpos[4]   # X-axis tilt
        pitch = self.data.qpos[5]   # Y-axis tilt

        orientation_penalty = -np.sum(np.square(self.data.qpos[3:5])) * 1.4
        # excess_pitch = max(0.0, abs(pitch) - 0.12)
        # pitch_penalty = -5.0 * (excess_pitch ** 2)
        excess_roll = max(0.0, abs(roll) - 0.15)     
        roll_penalty = -0.3 * (excess_roll ** 2)  

        # Survival rewards
        alive_bonus = 0.5
        survival_bonus = 0.1  # Small reward for each step survived

        # Lower control cost
        ctrl_cost = 0.00075 * np.square(action).sum()

        foot_geom_ids = {f: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f)
                 for f in ["FL", "FR", "RL", "RR"]}

        # Reset contact flags
        foot_contact = {f: False for f in foot_geom_ids}

        # Loop through contacts in this step
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in foot_geom_ids.values() or c.geom2 in foot_geom_ids.values():
                # mark whichever foot is in contact
                for f, gid in foot_geom_ids.items():
                    if c.geom1 == gid or c.geom2 == gid:
                        foot_contact[f] = True

        # Diagonal gait reward
        diag1 = foot_contact["FL"] and foot_contact["RR"]
        diag2 = foot_contact["FR"] and foot_contact["RL"]

        current_diag = None
        if diag1:
            current_diag = "diag1"
        elif diag2:
            current_diag = "diag2"

        # Track how long we stay in the same diagonal
        if current_diag is not None:
            if current_diag == self.last_diag:
                self.same_diag_count += 1
            else:
                self.same_diag_count = 1  # reset count when switching
            self.last_diag = current_diag
        else:
            self.same_diag_count = 0
            self.last_diag = None

        # Base gait reward
        gait_reward = 0.5 if current_diag else -0.2

        # Add penalty if stuck in the same diagonal too long
        if self.same_diag_count >= 40:
            gait_reward -= 1.0   # penalize strongly

        reward = (tracking_reward + 
                 height_bonus + 
                #  orientation_penalty + 
                 alive_bonus + 
                 survival_bonus - 
                 ctrl_cost +
                 joint_reg_penalty + 
                 gait_reward
                #  roll_penalty
                 )

        # Termination conditions (much more forgiving)
        height_termination = height < 0.15 or height > 0.8  # Wider bounds
        terminated = bool(height_termination)
        
        # Truncation after max steps
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