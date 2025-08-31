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

        # Enhanced gait policy parameters
        self.gait_frequency = 1.5      # Slower frequency for stability
        self.hip_amplitude = 0.3       # Reduced amplitude for hip joints
        self.knee_amplitude = 0.5      # Higher amplitude for knee joints
        
        # Trot gait phase offsets (diagonal pairs move together)
        # Assuming joint order: [fl_hip, fl_knee, fr_hip, fr_knee, rl_hip, rl_knee, rr_hip, rr_knee]
        if self.n_joints == 8:
            self.phase_offsets = np.array([0.0, 0.0, np.pi, np.pi, np.pi, np.pi, 0.0, 0.0])
            self.joint_amplitudes = np.array([self.hip_amplitude, self.knee_amplitude, 
                                            self.hip_amplitude, self.knee_amplitude,
                                            self.hip_amplitude, self.knee_amplitude, 
                                            self.hip_amplitude, self.knee_amplitude])
        else:
            # Fallback for different joint configurations
            self.phase_offsets = np.linspace(0, 2 * np.pi, self.n_joints, endpoint=False)
            self.joint_amplitudes = np.full(self.n_joints, 0.4)

        # self.joint_amplitudes[4:8] *= 0.5

        # Default joint positions for stability
        self.default_hip_angle = 0.1    # Slight hip flexion
        self.default_knee_angle = -0.8  # Bent knees for stability

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
        
        # Blend with more emphasis on gait policy for stability
        blend_ratio = 0.7  # 70% gait policy, 30% learned action
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

        # Compute roll & pitch
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        roll, pitch, _ = r.as_euler('xyz', degrees=False)

        # Enhanced reward components with better balance

        # 1) Forward velocity tracking (target 0.5 m/s)
        forward_vel = base_lin_vel[0]
        r_vel = np.exp(-((forward_vel - 0.5) ** 2) / 0.25)

        # 2) Height maintenance (target height 0.34 m) - increased importance
        height_error = (base_pos[2] - 0.34) ** 2
        r_height = np.exp(-height_error / 0.01)  # Tighter height control

        # 3) Enhanced posture stability 
        r_posture = np.exp(-((roll ** 2 + pitch ** 2)) / 0.02)  # Tighter posture control

        # 4) Joint position penalty to avoid extreme positions
        joint_positions = self.data.qpos[7:]
        r_joint = np.exp(-0.1 * np.sum(joint_positions ** 2))

        # 5) Action smoothness
        if not hasattr(self, 'prev_action'):
            self.prev_action = np.zeros_like(mixed_action)
        r_smooth = np.exp(-np.sum((mixed_action - self.prev_action) ** 2) / 0.1)
        self.prev_action = mixed_action.copy()

        # 6) Control cost
        r_ctrl = np.exp(-0.01 * np.sum(mixed_action ** 2))

        # 7) Alive bonus
        r_alive = 1.0

        # 8) Lateral stability (minimize sideways velocity)
        r_lateral = np.exp(-0.5 * (base_lin_vel[1] ** 2))

        # Rear-hip angles (model order: [fl_hip, fl_knee, fr_hip, fr_knee, rl_hip, rl_knee, rr_hip, rr_knee])
        rear_hips = self.data.qpos[7:][[4,6]] # indices 11 and 13 overall
        ref_angle = 0.1
        k_spread = 50.0
        r_spread = np.mean(np.exp(-k_spread * (rear_hips - ref_angle)**2))
        r_spread = np.clip(r_spread, 1e-3, 1.0)

        # Rebalanced reward
        reward = (
        0.20 * r_vel +
        0.25 * r_height +
        0.20 * r_posture +
        0.15 * r_joint +
        0.10 * r_spread +
        0.05 * r_smooth +
        0.01 * r_ctrl +    # reduced
        0.05 * r_alive +
        0.02 * r_lateral   # reduced
    )

        # Enhanced termination conditions
        terminated = True if (
            base_pos[2] < 0.25 or base_pos[2] > 0.6 or  # Tighter height bounds
            abs(roll) > 0.8 or abs(pitch) > 0.8 or      # Tighter angle bounds
            abs(base_lin_vel[1]) > 2.0                   # Prevent excessive sideways motion
        ) else False
        truncated = self.current_steps >= self.max_steps

        obs = self._get_obs()
        info = {
            'r_vel': r_vel,
            'r_height': r_height, 
            'r_posture': r_posture,
            'r_joint': r_joint,
            'gait_action': gait_action
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.current_steps = 0

        # Set more stable initial joint positions
        if self.n_joints == 8:
            # Hip joints slightly flexed
            self.data.qpos[[7, 9, 11, 13]] = self.default_hip_angle + np.random.uniform(-0.05, 0.05, 4)
            # Knee joints moderately bent
            self.data.qpos[[8, 10, 12, 14]] = self.default_knee_angle + np.random.uniform(-0.1, 0.1, 4)
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
