import gymnasium as gym
import numpy as np
import mujoco
from gymnasium import spaces


class Go2Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, xml_path="./unitree_go2/scene(friction).xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

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

        # Enhanced gait policy parameters - INCREASED for larger steps
        self.gait_frequency = 2.0      # Increased from 1.5 for faster stepping
        self.hip_amplitude = 0.5       # Increased from 0.3 for larger hip motion
        self.knee_amplitude = 0.8      # Increased from 0.5 for more knee lift
        
        # Trot gait phase offsets (diagonal pairs move together)
        if self.n_joints == 8:
            self.phase_offsets = np.array([0.0, 0.0, np.pi, np.pi, np.pi, np.pi, 0.0, 0.0])
            self.joint_amplitudes = np.array([self.hip_amplitude, self.knee_amplitude, 
                                            self.hip_amplitude, self.knee_amplitude,
                                            self.hip_amplitude, self.knee_amplitude, 
                                            self.hip_amplitude, self.knee_amplitude])
        else:
            self.phase_offsets = np.linspace(0, 2 * np.pi, self.n_joints, endpoint=False)
            self.joint_amplitudes = np.full(self.n_joints, 0.4)

        # Default joint positions for stability - ADJUSTED to prevent nosediving
        self.default_hip_angle = 0.05   # Reduced to prevent forward lean
        self.default_knee_angle = -0.6  # Less bent to allow larger steps

    def _get_obs(self):
        qpos = self.data.qpos[7:].ravel()
        qvel = self.data.qvel[6:].ravel()
        base_quat = self.data.qpos[3:7]
        base_linvel = self.data.qvel[0:3]
        base_angvel = self.data.qvel[3:6]
        return np.concatenate([qpos, qvel, base_quat, base_linvel, base_angvel]).astype(np.float32)

    def _compute_gait_action(self):
        t = self.current_steps / self.metadata["render_fps"]
        
        # Generate sinusoidal gait with default positions
        gait_pattern = np.sin(2 * np.pi * self.gait_frequency * t + self.phase_offsets)
        
        # Apply different amplitudes to different joints
        gait = self.joint_amplitudes * gait_pattern
        
        # Add default joint positions for stability
        if self.n_joints == 8:
            # Hip joints (indices 0, 2, 4, 6)
            gait[0::2] += self.default_hip_angle
            # Knee joints (indices 1, 3, 5, 7)  
            gait[1::2] += self.default_knee_angle
        
        return np.clip(gait, -1.0, 1.0)

    def step(self, action):
        # Compute gait action
        gait_action = self._compute_gait_action()
        
        # REDUCED blend ratio to allow more learned behavior
        blend_ratio = 0.5  # Reduced from 0.7 to allow larger steps
        mixed_action = blend_ratio * gait_action + (1 - blend_ratio) * action

        # Apply action scaling
        ctrl = self.action_low + (mixed_action + 1) * 0.5 * (self.action_high - self.action_low)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        self.current_steps += 1

        # Get state
        pos = self.data.qpos
        vel = self.data.qvel
        base_pos = pos[0:3]
        base_quat = pos[3:7]
        base_lin_vel = vel[0:3]
        base_ang_vel = vel[3:6]

        # Compute roll & pitch - FIXED TYPO HERE
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        roll, pitch, _ = r.as_euler('xyz', degrees=False)

        # REWARD RESTRUCTURE for larger steps and nosedive prevention

        # 1) Forward velocity tracking - INCREASED target for larger steps
        forward_vel = base_lin_vel[0]
        target_vel = 0.8  # Increased from 0.5
        r_vel = np.exp(-((forward_vel - target_vel) ** 2) / 0.5)  # Wider tolerance

        # 2) Height maintenance - RELAXED tolerance
        height_error = (base_pos[2] - 0.34) ** 2
        r_height = np.exp(-height_error / 0.05)  # Relaxed from 0.01

        # 3) ENHANCED pitch control to prevent nosediving
        pitch_penalty = 10.0 * pitch ** 2  # Strong pitch penalty
        r_posture = np.exp(-(roll ** 2 / 0.02 + pitch_penalty))

        # 4) Joint position penalty - REDUCED to allow larger motions
        r_joint = np.exp(-0.05 * np.sum(self.data.qpos[7:] ** 2))  # Reduced from 0.1

        # 5) Action smoothness - RELAXED to allow dynamic motion
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(mixed_action)
        r_smooth = np.exp(-np.sum((mixed_action - self.prev_action) ** 2) / 0.2)  # Relaxed
        self.prev_action = mixed_action.copy()

        # 6) Control cost - REDUCED to allow larger commands
        r_ctrl = np.exp(-0.005 * np.sum(mixed_action ** 2))  # Reduced from 0.01

        # 7) Alive bonus
        r_alive = 1.0

        # 8) Lateral stability
        r_lateral = np.exp(-0.5 * (base_lin_vel[1] ** 2))

        # 9) STEP LENGTH reward - NEW: encourages larger steps
        step_length = np.abs(forward_vel) * 0.016  # dt ≈ 1/60
        r_step_length = np.tanh(step_length / 0.02)  # Reward longer steps

        # 10) Rear-hip spread penalty - RELAXED
        rear_hips = self.data.qpos[7:][[4,6]]
        ref_angle = 0.05  # Reduced from 0.1
        k_spread = 20.0   # Reduced from 50.0
        r_spread = np.mean(np.exp(-k_spread * (rear_hips - ref_angle)**2))
        r_spread = np.clip(r_spread, 1e-3, 1.0)

        # REBALANCED reward emphasizing velocity and preventing nosedive
        reward = (
        0.32 * r_vel
        + 0.32 * r_posture
        + 0.10 * r_height
        + 0.025 * r_joint
        + 0.0175 * r_ctrl
        + 0.01 * r_smooth
        + 0.02 * r_lateral
        + 0.05 * r_alive
        + 0.12 * r_spread  # strong leg spread reward
        + 0.02 * r_step_length
    )


        # RELAXED termination conditions for learning larger gaits
        terminated = True if (
            base_pos[2] < 0.20 or base_pos[2] > 0.7 or   # Wider height bounds
            abs(roll) > 1.0 or abs(pitch) > 0.6 or       # Tighter pitch, looser roll
            abs(base_lin_vel[1]) > 2.5                    # Allow more lateral motion
        ) else False
        truncated = self.current_steps >= self.max_steps

        obs = self._get_obs()
        info = {
            'r_vel': r_vel,
            'r_height': r_height, 
            'r_posture': r_posture,
            'r_joint': r_joint,
            'r_step_length': r_step_length,
            'gait_action': gait_action
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_steps = 0

        # IMPROVED initial pose to prevent nosedive
        if self.n_joints == 8:
            # Hip joints - LESS forward lean
            self.data.qpos[[7, 9, 11, 13]] = self.default_hip_angle + np.random.uniform(-0.02, 0.02, 4)
            # Knee joints - LESS bent for larger steps
            self.data.qpos[[8, 10, 12, 14]] = self.default_knee_angle + np.random.uniform(-0.05, 0.05, 4)
        else:
            self.data.qpos[7:] += np.random.uniform(-0.01, 0.01, size=self.n_joints)
            
        self.data.qvel[:] += np.random.uniform(-0.01, 0.01, size=self.model.nv)
        
        if not hasattr(self, "default_dof_pos") or self.default_dof_pos is None:
            self.default_dof_pos = np.copy(self.data.qpos[7:])

        obs = self._get_obs()
        return obs, {}

    def render(self):
        if self.render_mode == "human":
            try:
                import mujoco.viewer
                if not hasattr(self, 'viewer') or self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                else:
                    self.viewer.sync()
            except (AttributeError, ImportError):
                raise ImportError(
                    "Rendering failed: mujoco.viewer not available and mujoco_py could not be imported. "
                    "Please install mujoco or ensure your environment supports rendering."
                )
