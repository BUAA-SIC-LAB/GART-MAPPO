# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils
import torch
import torch.distributions as D
from torch.func import vmap
import math
import numpy as np
import time

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv, List, Optional
from omni_drones.utils.torch import cpos, off_diag, others, make_cells, euler_to_quaternion, quat_axis
from omni_drones.robots.drone import MultirotorBase
from omni_drones.views import ArticulationView, RigidPrimView
from omni.isaac.core.prims import XFormPrimView
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec, BoundedTensorSpec

import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
import omni.isaac.core.objects as objects
from omni.isaac.debug_draw import _debug_draw
from pxr import UsdGeom, UsdPhysics, PhysxSchema

from omni_drones.controllers import (
    LeePositionController,
    AttitudeController,
    RateController
)

# from omni_drones.envs.platform.utils import create_frame

# Formation configurations - 以Y轴为对称轴，无人机编队横向分布，沿X轴飞行
# 编队配置：[x偏移, y偏移, z偏移]，x=0表示编队中心，y轴左右分布

TIGHT_FORMATION = [
    [0, 0, 0],      # 中心无人机
    [0, 1.2, 0],    # 右侧无人机 (+Y方向)
    [0, -1.2, 0],   # 左侧无人机 (-Y方向)
    [0, 2.4, 0],    # 右侧外围无人机
    [0, -2.4, 0],   # 左侧外围无人机
]

# 3架无人机的紧密编队 - 横向一字排开
TIGHT_FORMATION_3 = [
    [0, 0, 0],      # 中心无人机
    [0, 1.5, 0],    # 右侧无人机 (+Y方向)
    [0, -1.5, 0],   # 左侧无人机 (-Y方向)
]

# 5架无人机的宽松编队 - 横向分布更开
WIDE_FORMATION = [
    [0, 0, 0],      # 中心无人机
    [0, 2.0, 0],    # 右侧无人机
    [0, -2.0, 0],   # 左侧无人机
    [0, 4.0, 0],    # 右侧外围无人机
    [0, -4.0, 0],   # 左侧外围无人机
]

# V字型编队 - 以编队中心为顶点，向后展开
V_FORMATIO_5 = [
    [0, 0, 0],     # 领头无人机 (V字顶点)
    [1.2, 1.2, 0], # 右后无人机
    [1.2, -1.2, 0], # 左后无人机
    [2.4, 2.4, 0], # 右后外围无人机
    [2.4, -2.4, 0], # 左后外围无人机
]

# V字型编队 - 以编队中心为顶点，向后展开
V_FORMATION_3 = [
    [0, 0, 0],     # 领头无人机 (V字顶点)
    [-1.2, 1.2, 0], # 右后无人机
    [-1.2, -1.2, 0], # 左后无人机
]

# 雁形编队 - 斜线编队
WEDGE_FORMATION = [
    [0, 0, 0],     # 领头无人机
    [0.8, 1.0, 0], # 右后无人机
    [1.6, 2.0, 0], # 右后外围无人机
    [0.8, -1.0, 0], # 左后无人机
    [1.6, -2.0, 0], # 左后外围无人机
]

FORMATIONS = {
    "tight": TIGHT_FORMATION,
    "wide": WIDE_FORMATION,
    "v_shape": V_FORMATIO_5,
    "tight_3": TIGHT_FORMATION_3,  # 3架无人机选项
    "v_shape_3": V_FORMATION_3,  # 3架无人机的V字型编队
    "wedge": WEDGE_FORMATION,      # 雁形编队
}

def sample_from_grid(cells: torch.Tensor, n):
    idx = torch.randperm(cells.shape[0], device=cells.device)[:n]
    return cells[idx]

class FormationGateTraversal(IsaacEnv):
    """
    Multi-drone formation maintenance and dynamic gate traversal environment.
    
    This environment combines formation control with gate traversal tasks. Multiple drones 
    must maintain formation while navigating through a series of dynamic gates that move 
    and rotate in space.

    ## Observation

    - `obs_self`: The state of the drone (position, orientation, velocity)
    - `obs_others`: The relative states of other drones in the formation
    - `gate_info`: Information about the current target gate (position, orientation, size, velocity)
    - `formation_target`: Target formation relative positions
    - `time_encoding`: Time encoding for episode progress

    ## Reward

    - `formation`: Reward for maintaining formation shape
    - `gate_progress`: Reward for progressing towards and through gates
    - `endpoint_progress`: Reward for progressing towards final endpoint positions
    - `collision_avoidance`: Penalty for getting too close to other drones
    - `gate_traversal`: Bonus reward for successfully passing through gates

    The total reward combines formation maintenance, gate traversal, and endpoint progress.

    ## Episode End

    The episode terminates when:
    - Any drone crashes
    - Drones get too close to each other
    - Formation deviates too much from target
    - All gates are successfully traversed (success)

    ## Config
    """
    def __init__(self, cfg, headless):
        self.time_encoding = cfg.task.time_encoding
        self.safe_distance = cfg.task.safe_distance
        self.formation_tolerance = cfg.task.formation_tolerance
        self.gate_count = cfg.task.gate_count
        self.gate_spacing = cfg.task.gate_spacing
        # Support both old and new gate size parameters
        if hasattr(cfg.task, 'gate_width') and hasattr(cfg.task, 'gate_height'):
            self.gate_width = cfg.task.gate_width
            self.gate_height = cfg.task.gate_height
        else:
            # Fallback to old square gate_size parameter
            self.gate_width = cfg.task.gate_size
            self.gate_height = cfg.task.gate_size
        # self.gate_radius = cfg.task.gate_radius
        self.gate_movement_speed = cfg.task.gate_movement_speed
        self.gate_init_thetas = cfg.task.gate_init_thetas

        # Curriculum Learning Configuration
        self.curriculum_enable = getattr(cfg.task.curriculum_learning, 'enable', False)
        if self.curriculum_enable:
            self.stage_ratios = cfg.task.curriculum_learning.stage_ratios
            assert len(self.stage_ratios) == 3, "Must have exactly 3 stages"
            assert abs(sum(self.stage_ratios) - 1.0) < 1e-6, "Stage ratios must sum to 1.0"
            self.current_stage = 0  # 0: endpoint only, 1: formation, 2: formation + gates
            self.total_frames = 0
            self.current_global_frames = 0  # Track global frame count for transition
            
            # Smooth transition settings
            self.enable_smooth_transition = getattr(cfg.task.curriculum_learning, 'enable_smooth_transition', True)
            self.transition_frames = getattr(cfg.task.curriculum_learning, 'transition_frames', 5000000)
            self.transition_start_frame = -1  # -1 means not in transition
            self.transition_from_stage = -1
            self.transition_to_stage = -1
        else:
            self.current_stage = 2  # Default to full task
            self.enable_smooth_transition = False
        
        # Direct Learning Configuration (No Curriculum Learning)
        # Use default reward weights from config
        # self.velocity_reward_weight = cfg.task.velocity_reward_weight
        self.uprightness_reward_weight = cfg.task.uprightness_reward_weight
        # self.survival_reward_weight = cfg.task.survival_reward_weight
        # self.effort_reward_weight = cfg.task.effort_reward_weight
        self.formation_reward_weight = cfg.task.formation_reward_weight
        self.formation_cohesion_weight = cfg.task.formation_cohesion_weight
        
        # 新增 Laplacian 编队相关参数（参考 FormationUnified）
        self.formation_size_reward_weight = getattr(cfg.task, 'formation_size_reward_weight', 1.0)
        self.separation_penalty_weight = getattr(cfg.task, 'separation_penalty_weight', 2.0)

        self.eval = cfg.task.eval
        if self.eval:
            self.eval_data_log = []
        
        self.gate_progress_reward_weight = cfg.task.gate_progress_reward_weight
        self.gate_traversal_reward_weight = cfg.task.gate_traversal_reward_weight

        self.endpoint_progress_reward_weight = cfg.task.endpoint_progress_reward_weight
        
        self.collision_penalty_weight = cfg.task.collision_penalty_weight
        
        # 动作平滑性相关奖励权重（参考 FormationUnified）
        self.reward_action_smoothness_weight = getattr(cfg.task, 'reward_action_smoothness_weight', 1.0)
        self.reward_effort_weight = getattr(cfg.task, 'reward_effort_weight', 0.5)
        self.reward_throttle_smoothness_weight = getattr(cfg.task, 'reward_throttle_smoothness_weight', 2.0)
        self.reward_spin_weight = getattr(cfg.task, 'spin_reward_coeff', 1.0)
        self.heading_reward_weight = getattr(cfg.task, 'heading_reward_weight', 1.0)
        
        # Formation scale parameter for adjusting tightness
        self.formation_scale = getattr(cfg.task, 'formation_scale', 1.0)
        
        # Position parameters - 需要在 super().__init__() 之前定义，因为 _design_scene() 会用到
        self.start_x = cfg.task.start_x  # 起始X位置，现在在X轴负方向
        self.end_x = -cfg.task.start_x   # 终点X位置，在X轴正方向



        super().__init__(cfg, headless)

        self.init_rpy_dist = D.Uniform(
            torch.tensor([-0.05, -0.05, 0.], device=self.device) * torch.pi,
            torch.tensor([0.05, 0.05, 0.], device=self.device) * torch.pi
        )

        self.drone.initialize()
        self.init_poses = self.drone.get_world_poses(clone=True)

        # # Create view for kinematic control of gate movements across all environments
        # self.gate_view = XFormPrimView(
        #     prim_paths_expr="/World/envs/env_*/Gate_*",
        #     name="gate_view",
        #     reset_xform_properties=False
        # )
        # # Initialize gate view after scene creation
        # self.gate_view.initialize()

        # Initialize DebugDraw for trajectory visualization
        self.draw = _debug_draw.acquire_debug_draw_interface()
        
        # Trajectory visualization buffers
        self.drone_trajectories = []  # List of trajectory points for each drone
        self.trajectory_max_length = self.max_episode_length  # Maximum trajectory length to display
        self.visualization_enabled = True  # Flag to enable/disable visualization
        
        # Drone trajectory history for each environment and drone
        self.drone_trajectory_history = torch.zeros(
            self.num_envs, self.drone.n, self.trajectory_max_length, 3, device=self.device
        )
        self.trajectory_step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Formation state history for visualization (sample every N steps)
        self.formation_sample_interval = 30  # Sample formation every 30 steps
        self.formation_max_samples = self.trajectory_max_length // self.formation_sample_interval + 1
        self.formation_history = torch.zeros(
            self.num_envs, self.formation_max_samples, self.drone.n, 3, device=self.device
        )
        self.formation_sample_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Start and end position markers
        self.start_positions = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.end_positions = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)

        # 初始化摆动门的物理参数（参考Pendulum_v1实现）
        # 从配置文件中读取摆动门参数，如果没有配置则使用默认值
        self.pendulum_length = getattr(cfg.task, 'pendulum_length', 2.0)      # 摆臂长度 [米]
        self.pendulum_damping = getattr(cfg.task, 'pendulum_damping', 0.1)    # 阻尼系数
        self.pendulum_mass = getattr(cfg.task, 'pendulum_mass', 2.0)          # 门的质量 [kg]
        self.gravity = getattr(cfg.task, 'pendulum_gravity', 9.81)            # 重力加速度 [m/s²]
        
        # 摆动门状态跟踪张量
        # 门的摆动状态：[theta, dot_theta] - 角度和角速度
        self.gate_pendulum_state = torch.zeros(self.num_envs, self.gate_count, 2, device=self.device)  # [环境数, 门数, [theta, dot_theta]]
        
        # 门的固定支点位置（pivot point）- 门的摆动中心
        self.gate_pivot_points = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device)   # [环境数, 门数, XYZ坐标]
        
        # 门的当前中心位置（根据摆动状态计算得出）
        self.gate_positions = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device)      # 门中心位置 [环境数, 门数, XYZ坐标]
        self.gate_velocities = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device)     # 门中心速度 [环境数, 门数, XYZ速度]
        self.gate_rotations = torch.zeros(self.num_envs, self.gate_count, 4, device=self.device)      # 门旋转四元数 [环境数, 门数, XYZW四元数]
        self.gate_angular_velocities = torch.zeros(self.num_envs, self.gate_count, 3, device=self.device) # 门角速度 [环境数, 门数, XYZ角速度]
        
        # Gate traversal tracking
        self.current_gate_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gates_passed = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Per-drone gate traversal tracking: [num_envs, gate_count, n_drones]
        # True表示该无人机已经穿过该门
        self.drone_gate_passed = torch.zeros(
            self.num_envs, 
            self.gate_count, 
            self.drone.n,
            dtype=torch.bool, 
            device=self.device
        )
        
        # Reward tracking
        self.last_formation_error = torch.zeros(self.num_envs, device=self.device)
        # 修改为存储所有无人机到门的距离和（而非编队中心到门的距离）
        self.last_gate_distance = torch.zeros(self.num_envs, device=self.device)
        # Change to track individual drone distances to endpoints
        self.last_endpoint_distances = torch.zeros(self.num_envs, self.drone.n, device=self.device)

        # 生成起始位置的候选点 - 在起始区域内随机分布
        self.cells = (
            make_cells([self.start_x - 2, -3, 0.5], [self.start_x + 2, 3, 2.5], [0.8, 0.8, 0.4])
            .flatten(0, -2)
            .to(self.device)
        )

        # Formation and gate tracking
        self.formation_center_target = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Additional tracking variables for comprehensive stats
        self.prev_drone_velocities = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        self.prev_drone_positions = torch.zeros(self.num_envs, self.drone.n, 3, device=self.device)
        
        # 动作追踪变量（参考 FormationUnified）
        self.prev_actions = torch.zeros(self.num_envs, self.drone.n, 4, device=self.device)
        self.current_actions = torch.zeros(self.num_envs, self.drone.n, 4, device=self.device)
        
        self.gate_traversal_times = torch.zeros(self.num_envs, device=self.device)
        self.path_lengths = torch.zeros(self.num_envs, device=self.device)
        self.energy_consumption = torch.zeros(self.num_envs, device=self.device)
        self.near_collision_counter = torch.zeros(self.num_envs, device=self.device)
        self.ground_collision_counter = torch.zeros(self.num_envs, device=self.device)
        self.episode_start_time = torch.zeros(self.num_envs, device=self.device)
        self.behavior_history = torch.zeros(self.num_envs, 10, device=self.device)  # Track behavioral patterns
        
        # Consistency and learning progress tracking
        self.recent_rewards = torch.zeros(self.num_envs, 20, device=self.device)  # Track recent rewards for consistency
        self.reward_buffer_ptr = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.exploration_counter = torch.zeros(self.num_envs, device=self.device)
        self.novelty_buffer = torch.zeros(self.num_envs, 5, 3, device=self.device)  # Track position novelty
    
    def _pendulum_dynamics(self, state):
        """
        计算摆动门的动力学方程（参考Pendulum_v1实现）
        
        Args:
            state: 摆动状态 [batch_size, gate_count, 2] - [theta, dot_theta]
            
        Returns:
            state_dot: 状态导数 [batch_size, gate_count, 2] - [dot_theta, ddot_theta]
        """
        theta = state[..., 0]      # 摆动角度
        dot_theta = state[..., 1]  # 角速度
        
        # 摆动动力学方程：ddot_theta = -(g/L) * sin(theta) - (damping/mass) * dot_theta
        ddot_theta = -(self.gravity / self.pendulum_length) * torch.sin(theta) - \
                     (self.pendulum_damping / self.pendulum_mass) * dot_theta
        
        # 返回状态导数 [dot_theta, ddot_theta]
        return torch.stack([dot_theta, ddot_theta], dim=-1)
    
    def _get_gate_position_from_pendulum(self, pivot_point, theta):
        """
        根据摆动角度计算门的中心位置（参考Pendulum_v1实现）
        
        Args:
            pivot_point: 摆动支点位置 [batch_size, gate_count, 3]
            theta: 摆动角度 [batch_size, gate_count]
            
        Returns:
            position: 门中心位置 [batch_size, gate_count, 3]
        """
        pos = torch.zeros_like(pivot_point)
        pos[..., 0] = pivot_point[..., 0]  # X坐标保持不变
        pos[..., 1] = pivot_point[..., 1] + self.pendulum_length * torch.sin(theta)  # Y坐标变化
        pos[..., 2] = pivot_point[..., 2] - self.pendulum_length * torch.cos(theta)  # Z坐标变化
        return pos
    
    def _get_gate_velocity_from_pendulum(self, theta, dot_theta):
        """
        根据摆动角速度计算门的速度（参考Pendulum_v1实现）
        
        Args:
            theta: 摆动角度 [batch_size, gate_count] 
            dot_theta: 角速度 [batch_size, gate_count]
            
        Returns:
            velocity: 门中心速度 [batch_size, gate_count, 3]
        """
        vel = torch.zeros(theta.shape[0], theta.shape[1], 3, device=self.device)
        vel[..., 0] = 0.0  # X方向速度为0
        vel[..., 1] = self.pendulum_length * dot_theta * torch.cos(theta)  # Y方向速度
        vel[..., 2] = self.pendulum_length * dot_theta * torch.sin(theta)  # Z方向速度  
        return vel
    
    def _get_gate_quaternion_from_pendulum(self, theta):
        """
        根据摆动角度计算门的旋转四元数（参考Pendulum_v1实现）
        
        Args:
            theta: 摆动角度 [batch_size, gate_count]
            
        Returns:
            quaternion: 旋转四元数 [batch_size, gate_count, 4] - [x, y, z, w]
        """
        # 门绕X轴旋转theta角度
        half_theta = theta * 0.5
        quat = torch.zeros(theta.shape[0], theta.shape[1], 4, device=self.device)
        quat[..., 0] = torch.sin(half_theta)  # qx
        quat[..., 1] = 0.0                    # qy  
        quat[..., 2] = 0.0                    # qz
        quat[..., 3] = torch.cos(half_theta)  # qw
        return quat

    def _design_scene(self) -> Optional[List[str]]:
        drone_model_cfg = self.cfg.task.drone_model
        
        # 准备配置字典传递给 MultirotorBase.make
        # 将 OmegaConf 对象转换为字典以便访问
        drone_cfg_dict = {}
        if hasattr(drone_model_cfg, 'sim_mode'):
            drone_cfg_dict['sim_mode'] = drone_model_cfg.sim_mode
        
        self.drone, self.controller = MultirotorBase.make(
            drone_model_cfg.name, 
            drone_model_cfg.controller, 
            self.device,
            cfg=drone_cfg_dict if drone_cfg_dict else None
        )

        scene_utils.design_scene()

        # Set up formation
        formation = self.cfg.task.formation
        if isinstance(formation, str):
            self.formation = torch.as_tensor(
                FORMATIONS[formation], device=self.device
            ).float()
        elif isinstance(formation, list):
            self.formation = torch.as_tensor(
                formation, device=self.device
            ).float()
        else:
            raise ValueError(f"Invalid target formation {formation}")

        # Apply formation scale parameter
        self.formation = self.formation * self.formation_scale
        
        # Compute Laplacian matrices for formation cost (参考 FormationUnified)
        # 归一化拉普拉斯矩阵（尺度不变）
        self.formation_L = laplacian(self.formation, normalize=True)
        # 非归一化拉普拉斯矩阵（用于辅助尺度控制）
        self.formation_L_unnormalized = laplacian(self.formation, normalize=False)
        
        # 计算标准编队尺寸（最大成对距离）
        formation_distances = torch.cdist(self.formation, self.formation)
        self.standard_formation_size = formation_distances.max().item()

        # Spawn drones in formation at starting position
        # 设置无人机初始生成位置在起始区域
        start_pos = torch.tensor([self.start_x, 0.0, 2.0], device=self.device, dtype=torch.float32)
        spawn_positions = self.formation + start_pos
        self.drone.spawn(translations=spawn_positions)

        # Create dynamic gates
        # self._create_dynamic_gates()

        return ["/World/defaultGroundPlane"]

    def _set_specs(self):
        # drone_state_dim = self.drone.state_spec.shape[0]
        drone_state_dim = 13  # 位置(3) + 速度(3) + 四元数(4) + 角速度(3) = 13维
        obs_self_dim = drone_state_dim
        
        if self.time_encoding:
            self.time_encoding_dim = 4
            obs_self_dim += self.time_encoding_dim

        # Gate information: 4 corner positions (4*3=12) + linear velocity (3) + angular velocity (3) = 18 dimensions
        gate_info_dim = 4 * 3 + 3 + 3  # 18 dimensions
        # Gate information for central obs: 4 corner positions (4*3=12) + linear velocity (3) + angular velocity (3) = 18 dimensions  
        gate_central_dim = 4 * 3 + 3 + 3  # 18 dimensions
        # Formation target: position error from ideal formation position (3 dimensions)
        formation_target_dim = 3  # 改为3维：当前位置到理想位置的误差向量
        # Endpoint information: relative position to final target (3) + distance to endpoint (1) = 4 dimensions
        endpoint_info_dim = 4

        observation_spec = CompositeSpec({
            "obs_self": UnboundedContinuousTensorSpec((1, obs_self_dim)),
            "obs_others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim)),
            "gate_info": UnboundedContinuousTensorSpec((1, gate_info_dim)),
            "formation_target": UnboundedContinuousTensorSpec((1, formation_target_dim)),
            "endpoint_info": UnboundedContinuousTensorSpec((1, endpoint_info_dim)),
        }).to(self.device)
        
        observation_central_spec = CompositeSpec({
            "drones": UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)),
            "gates": UnboundedContinuousTensorSpec((self.gate_count, gate_central_dim)),
            "formation": UnboundedContinuousTensorSpec((self.drone.n, 3)),
        }).to(self.device)
        
        self.observation_spec = CompositeSpec({
            "agents": {
                "observation": observation_spec.expand(self.drone.n),
                "observation_central": observation_central_spec,
            }
        }).expand(self.num_envs).to(self.device)

        # {'agents': {'action': torch.Size([1024, 3, 4]),
        #             'observation': {'endpoint_info': torch.Size([1024, 3, 1, 4]),
        #                             'formation_target': torch.Size([1024, 3, 1, 3]),
        #                             'gate_info': torch.Size([1024, 3, 1, 18]),
        #                             'obs_others': torch.Size([1024, 3, 2, 23]),
        #                             'obs_self': torch.Size([1024, 3, 1, 23])},
        #             'observation_central': {'drones': torch.Size([1024, 3, 23]),
        #                                     'formation': torch.Size([1024, 3, 3]),
        #                                     'gates': torch.Size([1024, 1, 18])}},
        
        self.action_spec = CompositeSpec({
            "agents": {
                "action": torch.stack([self.drone.action_spec] * self.drone.n, dim=0),
            }
        }).expand(self.num_envs).to(self.device)
        
        self.reward_spec = CompositeSpec({
            "agents": {
                "reward": UnboundedContinuousTensorSpec((self.drone.n, 1))
            }
        }).expand(self.num_envs).to(self.device)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_key=("agents", "observation"),
            action_key=("agents","action"),
            reward_key=("agents", "reward"),
            state_key=("agents", "observation_central")
        )
        
        # Comprehensive stats for monitoring RL training progress
        stats_spec = CompositeSpec({
            # Basic episode metrics
            "return": UnboundedContinuousTensorSpec(self.drone.n),
            
            # Basic reward components
            "reward_uprightness": UnboundedContinuousTensorSpec(1),
            "reward_heading": UnboundedContinuousTensorSpec(1),
            
            # Smoothness reward components (matching FormationUnified)
            "reward_effort": UnboundedContinuousTensorSpec(1),
            "reward_throttle_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_action_smoothness": UnboundedContinuousTensorSpec(1),
            "reward_spin": UnboundedContinuousTensorSpec(1),
            
            # Formation reward components
            "reward_formation": UnboundedContinuousTensorSpec(1),
            "reward_size": UnboundedContinuousTensorSpec(1),
            "reward_separation": UnboundedContinuousTensorSpec(1),
            
            # Gate traversal reward components
            "reward_gate_progress": UnboundedContinuousTensorSpec(1),
            "reward_gate_traversal": UnboundedContinuousTensorSpec(1),
            "reward_bypass_gate_penalty": UnboundedContinuousTensorSpec(1),
            
            # Endpoint reward components
            "reward_endpoint_progress": UnboundedContinuousTensorSpec(1),
            
            # Safety reward components
            "reward_collision_penalty": UnboundedContinuousTensorSpec(1),
            
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["stats"] = stats_spec
        self.stats = stats_spec.zero()

    def _reset_idx(self, env_ids: torch.Tensor):
        self.drone._reset_idx(env_ids)

        # 设置无人机起始位置 - 在门的一侧（X轴负方向起点）
        # 起始位置基于编队配置和随机化
        n_envs = len(env_ids)
        
        # 基础起始位置：编队中心位于起始区域
        base_start_pos = torch.tensor([self.start_x, 0.0, 1], device=self.device, dtype=torch.float32)
        
        # 添加小量随机化
        pos_noise = torch.randn(n_envs, 3, device=self.device) 
        pos_noise[:, 0] = torch.clamp(pos_noise[:, 0], -0.05, 0.05)  # X方向噪声限制在±1米
        
        # 计算每个环境的编队中心位置
        formation_centers = base_start_pos + pos_noise

        formation_centers = base_start_pos.unsqueeze(0)

        # print("formation centers shape:", formation_centers.shape)
        
        # 应用编队配置偏移 (已经包含formation_scale)
        formation_offset = self.formation.unsqueeze(0).expand(n_envs, -1, -1)
        pos = formation_centers.unsqueeze(1) + formation_offset


        rpy = self.init_rpy_dist.sample((*env_ids.shape, 3, 1))
        rot = euler_to_quaternion(rpy)
        
        # 随机朝向（主要是yaw角度）
        # rpy = torch.zeros(n_envs, self.drone.n, 3, device=self.device)
        # # rpy[:, :, 2] = torch.rand(n_envs, self.drone.n, device=self.device) * 0.4 - 0.2  # yaw: ±0.2弧度
        # rot = euler_to_quaternion(rpy)
        
        # 设置无人机位置和朝向
        self.drone.set_world_poses(
            pos + self.envs_positions[env_ids].unsqueeze(1), rot, env_ids
        )
        self.drone.set_velocities(torch.zeros_like(self.drone.get_velocities()[env_ids]), env_ids)

        # 存储轨迹可视化的起点位置
        self.start_positions[env_ids] = pos + self.envs_positions[env_ids].unsqueeze(1)
        
        # 计算终点位置 - 在门的另一侧（X轴正方向终点）
        # 终点位置：最后一个门位置再往终点方向偏移1米
        if self.gate_count == 1:
            last_gate_x = (self.start_x + self.end_x) / 2  # 单个门在中间
        else:
            # 计算最后一个门的X位置，与_create_dynamic_gates中的逻辑保持一致
            total_flight_distance = abs(self.start_x - self.end_x)
            usable_distance = total_flight_distance - 2.0
            gate_spacing_calculated = usable_distance / (self.gate_count - 1)
            start_position = self.start_x + 1.0
            last_gate_x = start_position + (self.gate_count - 1) * gate_spacing_calculated
        
        # 终点位置：最后一个门再往终点方向偏移1米
        final_x = last_gate_x + abs(self.start_x)  # 向正方向再偏移1米

        
        # 终点编队中心
        end_center = torch.tensor([final_x, 0.0, 1], device=self.device)
        # 保持编队形状到达终点 (formation_offset已经包含scale)
        self.end_positions[env_ids] = (end_center.unsqueeze(0).unsqueeze(1) + formation_offset + 
                                     self.envs_positions[env_ids].unsqueeze(1))
        
        # 重置轨迹历史
        self.drone_trajectory_history[env_ids] = 0
        self.trajectory_step_count[env_ids] = 0
        
        # 重置编队历史
        self.formation_history[env_ids] = 0
        self.formation_sample_count[env_ids] = 0

        # Reset gate tracking
        self.current_gate_idx[env_ids] = 0
        self.gates_passed[env_ids] = 0
        
        # Reset per-drone gate traversal tracking
        self.drone_gate_passed[env_ids] = False
        
        # Reset gate dynamics
        self._reset_gate_dynamics(env_ids)
        
        # Reset reward tracking
        self.last_formation_error[env_ids] = 0
        
        # Initialize distance tracking with actual distances (not zero)
        # Get current drone positions after reset
        drone_pos, _ = self.drone.get_world_poses()
        formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
        
        # Initialize gate distance sum to first gate for each environment - 修改为所有无人机距离和

        # 获取所有重置环境的第一个门位置 [len(env_ids), 3]
        first_gate_positions = self.gate_positions[env_ids, 0]  
        # 计算每个无人机到第一个门的距离 [len(env_ids), n_drones]
        individual_gate_distances = torch.norm(drone_pos[env_ids] - first_gate_positions.unsqueeze(1), dim=-1)
        # 计算所有无人机到门距离的总和 [len(env_ids)]
        gate_distance_sums = individual_gate_distances.sum(dim=-1)
        self.last_gate_distance[env_ids] = gate_distance_sums

        
        # Initialize endpoint distances - 张量化版本，计算每个无人机到对应终点的距离
        # drone_pos shape: [len(env_ids), n_drones, 3]
        # self.end_positions shape: [num_envs, n_drones, 3]
        drone_pos_reset = drone_pos[env_ids]  # [len(env_ids), n_drones, 3]
        end_positions_reset = self.end_positions[env_ids]  # [len(env_ids), n_drones, 3]
        # 计算每个无人机到其对应终点的距离 [len(env_ids), n_drones]
        individual_endpoint_distances = torch.norm(drone_pos_reset - end_positions_reset, dim=-1)
        self.last_endpoint_distances[env_ids] = individual_endpoint_distances
        
        # Reset comprehensive stats
        # Basic episode metrics
        self.stats["return"][env_ids] = 0

        # Basic reward components
        self.stats["reward_uprightness"][env_ids] = 0
        self.stats["reward_heading"][env_ids] = 0
        
        # Smoothness reward components (matching FormationUnified)
        self.stats["reward_effort"][env_ids] = 0
        self.stats["reward_throttle_smoothness"][env_ids] = 0
        self.stats["reward_action_smoothness"][env_ids] = 0
        self.stats["reward_spin"][env_ids] = 0

        # Gate traversal reward components
        self.stats["reward_gate_progress"][env_ids] = 0
        self.stats["reward_gate_traversal"][env_ids] = 0
        self.stats["reward_bypass_gate_penalty"][env_ids] = 0
        
        # Formation reward components
        self.stats["reward_formation"][env_ids] = 0
        self.stats["reward_size"][env_ids] = 0
        self.stats["reward_separation"][env_ids] = 0
        
        # Endpoint reward components
        self.stats["reward_endpoint_progress"][env_ids] = 0
        
        # Safety reward components
        self.stats["reward_collision_penalty"][env_ids] = 0

        
        # Reset additional tracking variables
        self.prev_drone_velocities[env_ids] = 0
        # 初始化 prev_drone_positions 为当前无人机位置
        current_drone_pos, _ = self.drone.get_world_poses()
        self.prev_drone_positions[env_ids] = current_drone_pos[env_ids]
        self.prev_actions[env_ids] = 0
        self.gate_traversal_times[env_ids] = 0
        self.energy_consumption[env_ids] = 0
        self.near_collision_counter[env_ids] = 0
        self.ground_collision_counter[env_ids] = 0
        self.episode_start_time[env_ids] = self.progress_buf[env_ids].float()
        self.behavior_history[env_ids] = 0
        self.recent_rewards[env_ids] = 0
        self.reward_buffer_ptr[env_ids] = 0
        self.exploration_counter[env_ids] = 0
        self.novelty_buffer[env_ids] = 0

    def _reset_gate_dynamics(self, env_ids: torch.Tensor):
        """
        重置摆动门的动力学状态（参考Pendulum_v1实现）
        
        摆动门重置策略：
        - 设置门的固定支点位置（pivot points）
        - 初始化摆动状态：角度和角速度
        - 根据摆动状态计算门的中心位置和旋转
        
        Args:
            env_ids: 需要重置的环境ID列表
        """
        for i in range(self.gate_count):
            # 计算门的支点位置（摆动中心）
            total_flight_distance = abs(self.start_x - self.end_x)  # 从起点到终点的总距离
            usable_distance = total_flight_distance - 2.0  # 留出起点和终点的缓冲区域
            
            if self.gate_count == 1:
                pivot_x = (self.start_x + self.end_x) / 2  # 单个门放在中间
            else:
                gate_spacing_calculated = usable_distance / (self.gate_count - 1)
                start_position = self.start_x + 1.0  # 从起点偏移1米开始
                pivot_x = start_position + i * gate_spacing_calculated
            
            pivot_y = 0.0  # 支点在Y轴方向居中
            # 支点高度：地面偏移1米 + 摆臂长度（这样门在垂直时刚好在适当高度）
            pivot_z = 1.0 + self.pendulum_length
            
            # **修复：添加环境偏移**
            # 创建基础支点位置（相对于环境中心）
            base_pivot = torch.tensor(
                [pivot_x, pivot_y, pivot_z], 
                device=self.device, 
                dtype=torch.float32
            ).expand(len(env_ids), -1)  # [len(env_ids), 3]
            
            # 添加环境偏移，使每个环境的门位于正确位置
            self.gate_pivot_points[env_ids, i] = base_pivot + self.envs_positions[env_ids]
            
            # 初始化摆动状态：随机初始角度和角速度
            # theta: 摆动角度，0表示垂直向下，正值表示向右摆，负值表示向左摆
            # initial_theta = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.5  # ±0.25弧度 (约±14度)
            # initial_dot_theta = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.2  # ±0.1弧度/秒
            if self.gate_init_thetas == -1:
                initial_theta = (torch.rand(len(env_ids), device=self.device) - 0.5) * torch.pi 
            else:
                initial_theta = torch.full((len(env_ids),), self.gate_init_thetas/180.0 * torch.pi, device=self.device)
            
            initial_dot_theta = torch.zeros(len(env_ids), device=self.device)

            self.gate_pendulum_state[env_ids, i, 0] = initial_theta      # 角度
            self.gate_pendulum_state[env_ids, i, 1] = initial_dot_theta  # 角速度
            
            # 根据摆动状态计算门的初始位置、速度和旋转
            pivot_points = self.gate_pivot_points[env_ids, i].unsqueeze(1)  # [num_envs, 1, 3]
            theta = self.gate_pendulum_state[env_ids, i, 0].unsqueeze(1)   # [num_envs, 1]
            dot_theta = self.gate_pendulum_state[env_ids, i, 1].unsqueeze(1)  # [num_envs, 1]
            
            # 计算门的中心位置
            self.gate_positions[env_ids, i] = self._get_gate_position_from_pendulum(
                pivot_points, theta
            ).squeeze(1)  # [num_envs, 3]
            
            # 计算门的速度
            self.gate_velocities[env_ids, i] = self._get_gate_velocity_from_pendulum(
                theta, dot_theta
            ).squeeze(1)  # [num_envs, 3]
            
            # 计算门的旋转
            self.gate_rotations[env_ids, i] = self._get_gate_quaternion_from_pendulum(
                theta
            ).squeeze(1)  # [num_envs, 4]
            
            # 设置门的初始角速度（主要是绕X轴的角速度，对应摆动的dot_theta）
            self.gate_angular_velocities[env_ids, i, 0] = initial_dot_theta  # X轴角速度 = 摆动角速度
            self.gate_angular_velocities[env_ids, i, 1] = 0.0               # Y轴角速度 = 0
            self.gate_angular_velocities[env_ids, i, 2] = 0.0               # Z轴角速度 = 0
        
        # Set initial gate positions using the view
        # if hasattr(self, 'gate_view') and self.gate_view is not None:
        #     try:
        #         # Prepare positions and orientations for reset environments
        #         reset_positions = self.gate_positions[env_ids].reshape(-1, 3)  # [len(env_ids) * gate_count, 3]
        #         reset_orientations = self.gate_rotations[env_ids].reshape(-1, 4)  # [len(env_ids) * gate_count, 4]
                
        #         # Create environment indices for gates in reset environments
        #         reset_env_indices = torch.repeat_interleave(
        #             env_ids,
        #             self.gate_count
        #         )
                
        #         # Update gate positions using the view
        #         self.gate_view.set_world_poses(
        #             positions=reset_positions,
        #             orientations=reset_orientations,
        #             env_indices=reset_env_indices
        #         )
                
        #     except Exception as e:
        #         # Fallback to direct prim manipulation only for env_0 if it's being reset
        #         if 0 in env_ids:
        #             for i in range(self.gate_count):
        #                 gate_path = f"/World/envs/env_0/Gate_{i}"
        #                 if prim_utils.is_prim_path_valid(gate_path):
        #                     gate_pos = self.gate_positions[0, i].cpu().numpy()
        #                     # Set position using USD
        #                     prim = prim_utils.get_prim_at_path(gate_path)
        #                     if prim:
        #                         xform = UsdGeom.Xformable(prim)
        #                         if xform:
        #                             xform.ClearXformOpOrder()
        #                             translate_op = xform.AddTranslateOp()
        #                             translate_op.Set(tuple(gate_pos))

    def _pre_sim_step(self, tensordict: TensorDictBase):
        # time.sleep(0.1)
        actions = tensordict[("agents", "action")]
        # print("actions:", actions)
        # Store actions for stats calculations
        self.actions = actions.clone()
        
        # 保存动作用于动作平滑奖励计算 (参考 FormationUnified)
        self.prev_actions = self.current_actions.clone()
        self.current_actions = actions.clone()
        
        self._update_gate_dynamics()

        # Get current drone state
        drone_state = self.drone.get_state()[..., :13]
        
        # Use the controller to convert high-level actions to rotor commands
        # Depending on controller type, process the actions accordingly
        if isinstance(self.controller, LeePositionController):
            # For position controller: actions are [target_pos, target_yaw]
            current_pos, _ = self.drone.get_world_poses()
            target_pos = actions[..., :3] + current_pos  # Relative position target
            target_yaw = actions[..., 3:4] * torch.pi  # Scale to radians
            rotor_commands = self.controller.compute(
                drone_state, 
                target_pos=target_pos,
                target_yaw=target_yaw
            )
        elif isinstance(self.controller, AttitudeController):
            # For attitude controller: actions are [thrust, yaw_rate, roll, pitch]
            target_thrust = ((actions[..., 0:1] + 1) / 2).clip(0.) * self.controller.max_thrusts.sum(-1)
            target_yaw_rate = actions[..., 1:2] * torch.pi
            target_roll = actions[..., 2:3] * torch.pi/4  # Scale to reasonable range
            target_pitch = actions[..., 3:4] * torch.pi/4  # Scale to reasonable range
            rotor_commands = self.controller(
                drone_state,
                target_thrust=target_thrust,
                target_yaw_rate=target_yaw_rate,
                target_roll=target_roll,
                target_pitch=target_pitch
            )
        elif isinstance(self.controller, RateController):
            # For rate controller: actions are [rate_x, rate_y, rate_z, thrust]
            target_rate = actions[..., :3] * torch.pi  # Scale to radians
            target_thrust = ((actions[..., 3:4] + 1) / 2).clip(0.) * self.controller.max_thrusts.sum(-1)
            # target_rate = actions[..., 1:]
            # target_thrust = actions[..., 0]

            # print("target_rate:", target_rate)
            # print("target_thrust:", target_thrust)
            
            # 根据仿真模式调用不同的方法
            if self.drone.sim_mode == "kinematic":
                # Kinematic 模式：直接传递物理控制量
                self.effort = self.drone.apply_action(None, target_rate=target_rate, target_thrust=target_thrust)
            else:
                # Physics 模式：通过 RateController 转换为电机指令
                rotor_commands = self.controller(
                    drone_state,
                    target_rate=target_rate,
                    target_thrust=target_thrust
                )
                torch.nan_to_num_(rotor_commands, 0.)
                self.effort = self.drone.apply_action(rotor_commands)
        else:
            # Default: pass through actions directly
            rotor_commands = actions
            torch.nan_to_num_(rotor_commands, 0.)
            self.effort = self.drone.apply_action(rotor_commands)


    def _update_gate_dynamics(self):
        """
        使用摆动物理更新门的动态状态（参考Pendulum_v1实现）
        
        摆动门的物理更新：
        1. 使用Runge-Kutta 4阶方法数值积分摆动方程
        2. 根据更新后的摆动状态计算门的位置、速度和旋转
        3. 确保门的运动符合真实的摆动物理
        """
        dt = self.dt  # 时间步长
        
        # 对所有环境的所有门同时进行物理更新
        # 使用Runge-Kutta 4阶方法进行数值积分（分4个子步骤获得更高精度）
        M = 4  # 子步骤数
        sub_dt = dt / M  # 子步骤时间间隔
        
        current_state = self.gate_pendulum_state.clone()  # [num_envs, gate_count, 2]
        
        for _ in range(M):
            # RK4积分的四个步骤
            k1 = sub_dt * self._pendulum_dynamics(current_state)
            k2 = sub_dt * self._pendulum_dynamics(current_state + 0.5 * k1)
            k3 = sub_dt * self._pendulum_dynamics(current_state + 0.5 * k2)
            k4 = sub_dt * self._pendulum_dynamics(current_state + k3)
            
            # 更新状态
            current_state = current_state + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        
        # 保存更新后的摆动状态
        self.gate_pendulum_state = current_state
        
        # 根据新的摆动状态计算门的物理属性
        for i in range(self.gate_count):
            theta = self.gate_pendulum_state[:, i, 0]      # 当前摆动角度
            dot_theta = self.gate_pendulum_state[:, i, 1]  # 当前角速度
            
            # 计算门的中心位置
            pivot_points = self.gate_pivot_points[:, i].unsqueeze(1)  # [num_envs, 1, 3]
            theta_expanded = theta.unsqueeze(1)  # [num_envs, 1]
            dot_theta_expanded = dot_theta.unsqueeze(1)  # [num_envs, 1]
            
            self.gate_positions[:, i] = self._get_gate_position_from_pendulum(
                pivot_points, theta_expanded
            ).squeeze(1)
            
            # 计算门的速度
            self.gate_velocities[:, i] = self._get_gate_velocity_from_pendulum(
                theta_expanded, dot_theta_expanded
            ).squeeze(1)
            
            # 计算门的旋转
            self.gate_rotations[:, i] = self._get_gate_quaternion_from_pendulum(
                theta_expanded
            ).squeeze(1)
            
            # 计算门的角速度 (主要是绕X轴的角速度，对应摆动的dot_theta)
            self.gate_angular_velocities[:, i, 0] = dot_theta  # X轴角速度 = 摆动角速度
            self.gate_angular_velocities[:, i, 1] = 0.0        # Y轴角速度 = 0
            self.gate_angular_velocities[:, i, 2] = 0.0        # Z轴角速度 = 0
        
        # # Apply positions and rotations to all gate objects using the view
        # if hasattr(self, 'gate_view') and self.gate_view is not None:
        #     # Get the central environment index for visualization
        #     central_env_idx = self.num_envs // 2 if self.num_envs > 1 else 0
            
        #     for i in range(self.gate_count):
        #         gate_path = f"/World/envs/env_0/Gate_{i}"
        #         if prim_utils.is_prim_path_valid(gate_path):
        #             prim = prim_utils.get_prim_at_path(gate_path)
        #             if prim:
        #                 gate_pos = self.gate_positions[central_env_idx, i].cpu().numpy()
        #                 gate_rot = self.gate_rotations[central_env_idx, i].cpu().numpy()  # [qx, qy, qz, qw]
                        
        #                 # Set transform using USD
        #                 xform = UsdGeom.Xformable(prim)
        #                 if xform:
        #                     # Clear previous transforms and set new position
        #                     xform.ClearXformOpOrder()
        #                     translate_op = xform.AddTranslateOp()
        #                     translate_op.Set(tuple(gate_pos))
                            
        #                     # 设置摆动门的旋转（绕X轴旋转）
        #                     # gate_rot是[qx, qy, qz, qw]格式，对于绕X轴旋转，qx包含旋转信息
        #                     if abs(gate_rot[0]) > 1e-6:  # 检查X轴旋转分量是否显著
        #                         # 从四元数提取X轴旋转角度：angle = 2 * arcsin(qx) for rotation around X-axis
        #                         rotation_angle_rad = 2.0 * math.asin(abs(gate_rot[0]))
        #                         if gate_rot[0] < 0:  # 处理负角度
        #                             rotation_angle_rad = -rotation_angle_rad
        #                         rotation_angle_deg = math.degrees(rotation_angle_rad)
                                
        #                         rotate_op = xform.AddRotateXOp()  # 绕X轴旋转
        #                         rotate_op.Set(rotation_angle_deg)



    def _compute_state_and_obs(self):
        obs = self._compute_obs()
        
        # Update trajectory visualization
        self._update_trajectory_visualization()
        
        self._tensordict.update(obs)
        return self._tensordict

    def _compute_gate_corners(self, gate_pos, gate_rot, relative_to_pos=None):
        """
        Compute the 4 corner positions of a gate.
        
        Args:
            gate_pos: Gate center position [batch_size, 3]
            gate_rot: Gate rotation quaternion [batch_size, 4] 
            relative_to_pos: If provided, return positions relative to this point [batch_size, 3]
            
        Returns:
            corners: [batch_size, 4, 3] - 4 corner positions
        """
        batch_size = gate_pos.shape[0]
        half_w = self.gate_width * 0.5
        half_h = self.gate_height * 0.5
        
        # Define local corner offsets (gate frame: X forward, Y right, Z up)
        local_corners = torch.tensor([
            [0.0, -half_w, -half_h],  # bottom-left
            [0.0, -half_w,  half_h],  # top-left
            [0.0,  half_w, -half_h],  # bottom-right
            [0.0,  half_w,  half_h],  # top-right
        ], device=self.device, dtype=torch.float32)  # [4, 3]
        
        # Convert quaternion to rotation matrix (vectorized)
        # gate_rot: [batch_size, 4] - quaternion (x, y, z, w)
        x, y, z, w = gate_rot[:, 0], gate_rot[:, 1], gate_rot[:, 2], gate_rot[:, 3]  # [batch_size] each
        
        # Compute rotation matrices for all quaternions at once [batch_size, 3, 3]
        R = torch.stack([
            torch.stack([1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)], dim=1),
            torch.stack([  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)], dim=1),
            torch.stack([  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)], dim=1),
        ], dim=1)  # [batch_size, 3, 3]
        
        # Transform corners to world frame (vectorized)
        # local_corners: [4, 3], R: [batch_size, 3, 3]
        # Use einsum for batch matrix multiplication: [batch_size, 4, 3] = [batch_size, 3, 3] @ [4, 3].T
        corners_world = torch.einsum('bij,kj->bki', R, local_corners) + gate_pos.unsqueeze(1)  # [batch_size, 4, 3]
        
        # Make relative if requested
        if relative_to_pos is not None:
            corners_world = corners_world - relative_to_pos.unsqueeze(1)  # [batch_size, 4, 3]
            
        return corners_world  # [batch_size, 4, 3]

    def _compute_obs(self):
        # 获取无人机的世界坐标（包含环境偏移）
        drone_pos_world, drone_rot = self.drone.get_world_poses()
        drone_vel = self.drone.get_velocities()
        
        # **关键修改：移除环境偏移，获得相对于环境中心的坐标**
        # 这样所有环境的无人机都使用统一的坐标系
        drone_pos = drone_pos_world - self.envs_positions.unsqueeze(1)  # [num_envs, n_drones, 3]
        
        # 构建不包含环境偏移的drone_state
        # drone_state原本是 [位置(3) + 速度(3) + 四元数(4) + 角速度(3)] = 13维
        drone_state = torch.cat([
            drone_pos,  # 位置：不包含环境偏移
            drone_vel[..., :3],  # 线速度
            drone_rot,  # 四元数
            drone_vel[..., 3:]  # 角速度
        ], dim=-1)  # [num_envs, n_drones, 13]
        
        # 获取当前目标门的信息 - 改为18维：4个角点(12)+线速度(3)+角速度(3) - 向量化版本
        current_gates = self.current_gate_idx
        gate_info = torch.zeros(self.num_envs, 1, 18, device=self.device)  # 新门信息：18维特征
        
        # 计算编队中心（不包含环境偏移）
        formation_center = drone_pos.mean(dim=1, keepdim=True)  # [num_envs, 1, 3]
        
        # 向量化处理：为有效门计算信息
        valid_mask = current_gates < self.gate_count  # [num_envs] - 哪些环境有有效的门
        
        if valid_mask.any():
            valid_env_indices = torch.where(valid_mask)[0]  # 有效环境的索引
            valid_gate_indices = current_gates[valid_mask]  # 对应的门索引
            
            # 批量提取门的状态信息（世界坐标，包含环境偏移）
            gate_pos_world_batch = self.gate_positions[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            gate_rot_batch = self.gate_rotations[valid_env_indices, valid_gate_indices]  # [valid_envs, 4]
            gate_lin_vel_batch = self.gate_velocities[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            gate_ang_vel_batch = self.gate_angular_velocities[valid_env_indices, valid_gate_indices]  # [valid_envs, 3]
            
            # **关键修改：移除环境偏移，获得门的相对坐标**
            gate_pos_batch = gate_pos_world_batch - self.envs_positions[valid_env_indices]  # [valid_envs, 3]
            
            # 批量计算4个角点位置（使用相对坐标）
            corners_batch = self._compute_gate_corners(gate_pos_batch, gate_rot_batch)  # [valid_envs, 4, 3]
            corners_flat_batch = corners_batch.reshape(len(valid_env_indices), -1)  # [valid_envs, 12]
            
            # 批量组装门信息：角点(12) + 线速度(3) + 角速度(3) = 18维
            gate_info_batch = torch.cat([
                corners_flat_batch,    # 4个角点绝对坐标（不含环境偏移）(12维)
                gate_lin_vel_batch,    # 线速度 (3维)
                gate_ang_vel_batch     # 角速度 (3维)
            ], dim=-1)  # [valid_envs, 18]
            
            # 将批量计算的结果写入对应位置
            gate_info[valid_env_indices, 0] = gate_info_batch
        
        # Compute formation targets（不包含环境偏移）
        target_formation_global = self.formation.unsqueeze(0).expand(self.num_envs, -1, -1) + formation_center
        
        # 计算终点信息 - 向量化版本（不包含环境偏移）
        endpoint_info = torch.zeros(self.num_envs, 1, 4, device=self.device)  # 终点信息：4维特征
        
        # **关键修改：移除环境偏移后的终点位置**
        end_positions_local = self.end_positions - self.envs_positions.unsqueeze(1)  # [num_envs, n_drones, 3]
        end_center = end_positions_local.mean(dim=1)  # [num_envs, 3]
        center_pos = formation_center[:, 0]  # 当前编队中心 [num_envs, 3]
        
        # 批量计算相对位置向量 (3维)
        relative_end_pos = end_center - center_pos  # [num_envs, 3]
        
        # 批量计算到终点的距离 (1维)
        distance_to_endpoint = torch.norm(relative_end_pos, dim=-1)  # [num_envs]
        
        # 批量组装终点信息：相对位置(3) + 距离(1) = 4维
        endpoint_info[:, 0, :3] = relative_end_pos  # 相对终点位置 (3维): [dx, dy, dz]
        endpoint_info[:, 0, 3] = distance_to_endpoint  # 终点距离 (1维)
        
        # Individual agent observations
        obs_self_list = []
        obs_others_list = []
        gate_info_list = []
        formation_target_list = []
        endpoint_info_list = []
        
        for i in range(self.drone.n):
            # Self observation（已经是不含环境偏移的坐标）
            obs_self = drone_state[:, i:i+1]
            
            
            # Others observation（已经是不含环境偏移的坐标）
            other_indices = [j for j in range(self.drone.n) if j != i]
            obs_others = drone_state[:, other_indices].clone()
            
            # **关键修改：gate_info已经是绝对坐标（不含环境偏移），不需要转换为相对坐标**
            # 所有智能体看到的都是相同的门信息
            gate_info_agent = gate_info.clone()
            
            # Formation target: current position error from ideal formation position
            # （已经是不含环境偏移的坐标）
            formation_target = drone_pos[:, i] - target_formation_global[:, i]  # [num_envs, 3]
            formation_target_flat = formation_target.unsqueeze(1)  # [num_envs, 1, 3]
            
            obs_self_list.append(obs_self)
            obs_others_list.append(obs_others)
            gate_info_list.append(gate_info_agent)
            formation_target_list.append(formation_target_flat)
            endpoint_info_list.append(endpoint_info)
        
        # print("endpoint_info list:", endpoint_info_list)
        # print("formation_target list:", formation_target_list)
        # print("gate info list:", gate_info_list)
        # print("obs others list:", obs_others_list)
        # print("obs self list:", obs_self_list)

        # Stack observations to create proper tensor structure
        obs_self_stacked = torch.stack(obs_self_list, dim=1)  # [num_envs, num_agents, 1, obs_dim]
        obs_others_stacked = torch.stack(obs_others_list, dim=1)  # [num_envs, num_agents, num_others, obs_dim]
        gate_info_stacked = torch.stack(gate_info_list, dim=1)  # [num_envs, num_agents, 1, gate_dim]
        formation_target_stacked = torch.stack(formation_target_list, dim=1)  # [num_envs, num_agents, 1, formation_dim]
        endpoint_info_stacked = torch.stack(endpoint_info_list, dim=1)  # [num_envs, num_agents, 1, endpoint_dim]
        
        # **关键修改：中心化观测也使用不含环境偏移的坐标**
        # Central observation - 使用18维门信息：4个绝对角点(12)+线速度(3)+角速度(3) - 向量化版本
        all_gate_info = torch.zeros(self.num_envs, self.gate_count, 18, device=self.device)
        
        # 批量处理所有门的信息（世界坐标，包含环境偏移）
        gate_pos_world_all = self.gate_positions  # [num_envs, gate_count, 3]
        gate_rot_all = self.gate_rotations  # [num_envs, gate_count, 4]
        gate_lin_vel_all = self.gate_velocities  # [num_envs, gate_count, 3]
        gate_ang_vel_all = self.gate_angular_velocities  # [num_envs, gate_count, 3]
        
        # **关键修改：移除环境偏移**
        # 将世界坐标转换为相对于环境中心的坐标
        env_offsets_expanded = self.envs_positions.unsqueeze(1).expand(-1, self.gate_count, -1)  # [num_envs, gate_count, 3]
        gate_pos_all = gate_pos_world_all - env_offsets_expanded  # [num_envs, gate_count, 3]
        
        # 重塑数据以进行批量计算：将 [num_envs, gate_count, ...] 变为 [num_envs * gate_count, ...]
        batch_size = self.num_envs * self.gate_count
        gate_pos_flat = gate_pos_all.reshape(batch_size, 3)
        gate_rot_flat = gate_rot_all.reshape(batch_size, 4)
        gate_lin_vel_flat = gate_lin_vel_all.reshape(batch_size, 3)
        gate_ang_vel_flat = gate_ang_vel_all.reshape(batch_size, 3)
        
        # 批量计算所有门的4个角点绝对坐标（不含环境偏移）
        corners_all = self._compute_gate_corners(gate_pos_flat, gate_rot_flat)  # [batch_size, 4, 3]
        corners_flat_all = corners_all.reshape(batch_size, -1)  # [batch_size, 12]
        
        # 重塑回原始维度
        corners_reshaped = corners_flat_all.reshape(self.num_envs, self.gate_count, 12)
        
        # 批量组装门信息：角点(12) + 线速度(3) + 角速度(3) = 18维
        all_gate_info[:, :, :12] = corners_reshaped      # 4个角点绝对坐标（不含环境偏移）(12维)
        all_gate_info[:, :, 12:15] = gate_lin_vel_all    # 线速度 (3维)
        all_gate_info[:, :, 15:18] = gate_ang_vel_all    # 角速度 (3维)
        
        obs_central = {
            "drones": drone_state,  # 已经是不含环境偏移的坐标
            "gates": all_gate_info,  # 已经是不含环境偏移的坐标
            "formation": target_formation_global,  # 已经是不含环境偏移的坐标
        }
        
        return TensorDict({
            "agents": {
                "observation": {
                    "obs_self": obs_self_stacked,
                    "obs_others": obs_others_stacked,
                    "gate_info": gate_info_stacked,
                    "formation_target": formation_target_stacked,
                    "endpoint_info": endpoint_info_stacked,
                },
                "observation_central": obs_central,
            },
            "stats": self.stats.clone(),
        }, batch_size=self.batch_size)

    def _compute_reward_and_done(self):
        drone_pos, drone_rot = self.drone.get_world_poses()
        drone_vel = self.drone.get_velocities()
        
        # Get current curriculum stage
        current_stage = self.get_curriculum_stage()
        
        # ====================
        # CURRICULUM LEARNING REWARD FUNCTION
        # Different reward functions for different stages:
        # Stage 0: Endpoint only (basic flight)
        # Stage 1: Formation maintenance + Endpoint
        # Stage 2: Formation + Gates + Endpoint (full task)
        # ====================
        
        # 1. Basic Flight Stability Rewards (inspired by Forest.py)
        
        # 1.1 Velocity toward current objective reward (gate first, then endpoint)
        end_center = self.end_positions.mean(dim=1)  # [num_envs, 3]
        formation_center = drone_pos.mean(dim=1)  # [num_envs, 3]
        
        # Determine current objective: gate if available, otherwise endpoint (vectorized)
        # Check which environments have valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count  # [num_envs]
        
        # Get current gate positions for all environments using advanced indexing
        # For environments with valid gates, use gate position; for others, use endpoint
        env_indices = torch.arange(self.num_envs, device=self.device)
        # [Modified] Clamp index to ensure validity for distance calculation
        current_gate_idx_safe = self.current_gate_idx.clamp(0, self.gate_count-1)
        gate_positions_current = self.gate_positions[env_indices, current_gate_idx_safe]  # [num_envs, 3]
        
        # Compute heading reward - drone's forward direction should align with direction to target
        # Stage-aware target selection:
        # - Stage 0: Always use endpoint (focus on basic navigation)
        # - Stage 1: Always use endpoint (formation flying to endpoint)
        # - Stage 2: Use gate if not passed yet, otherwise endpoint (full task with gates)
        current_stage = self.get_curriculum_stage()
        
        if current_stage < 2:
            # Stage 0 and 1: Always target each drone's own endpoint
            # self.end_positions shape: [num_envs, n_drones, 3]
            target_points = self.end_positions  # [num_envs, n_drones, 3]
        else:
            # Stage 2: Individual drone targeting based on gate passage status
            # - If drone hasn't passed the current gate: target the gate
            # - If drone has passed the current gate: target its own endpoint
            
            # Get current gate index for each environment
            # self.drone_gate_passed shape: [num_envs, gate_count, n_drones]
            # current_gate_idx shape: [num_envs]
            
            # Initialize target points with endpoints
            target_points = self.end_positions.clone()  # [num_envs, n_drones, 3]
            
            # For environments with valid gates, vectorized version
            if valid_gate_mask.any():
                # Get gate passage status for current gate: [num_envs, n_drones]
                # Use advanced indexing to get the status for each environment's current gate
                env_indices = torch.arange(self.num_envs, device=self.device)
                current_gate_passed = self.drone_gate_passed[env_indices, current_gate_idx_safe]  # [num_envs, n_drones]
                
                # Create gate target for all drones: [num_envs, n_drones, 3]
                gate_target = gate_positions_current.unsqueeze(1).expand(-1, self.drone.n, -1)
                
                # For each drone: if gate is valid AND drone hasn't passed, use gate; otherwise use endpoint
                # drone_should_target_gate shape: [num_envs, n_drones]
                drone_should_target_gate = valid_gate_mask.unsqueeze(-1) & (~current_gate_passed)
                
                # Update target points using vectorized operation
                # drone_should_target_gate.unsqueeze(-1) shape: [num_envs, n_drones, 1]
                target_points = torch.where(
                    drone_should_target_gate.unsqueeze(-1),
                    gate_target,
                    self.end_positions
                )
        
        # Get direction from each drone to its target
        target_vectors = target_points - drone_pos  # [num_envs, n_drones, 3]
        target_dist = target_vectors.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [num_envs, n_drones, 1]
        target_directions = target_vectors / target_dist  # Normalized target direction
        
        # Get drone forward direction from quaternion (axis=0 is forward)
        drone_forward = quat_axis(drone_rot, axis=0)  # [num_envs, n_drones, 3]
        
        # Compute heading error as distance between forward direction and target direction
        heading_error = torch.norm(target_directions - drone_forward, dim=-1)  # [num_envs, n_drones]
        
        # Convert to reward (clipped to [0, 1])
        reward_heading = torch.clamp(1.0 - heading_error, min=0.0).mean(dim=-1) * self.heading_reward_weight  # [num_envs]
        
        
        # 1.2 Uprightness reward - maintaining stable orientation
        drone_up = quat_axis(drone_rot, axis=2)[..., 2].mean(dim=-1)  # Average Z-component
        reward_uprightness = torch.square((drone_up + 1) / 2) * self.uprightness_reward_weight
        
        # 1.3 综合动作平滑奖励 - 参考 FormationUnified 的完整实现
        # 包含4个组成部分：effort, throttle_smoothness, action_smoothness, spin
        
        # (a) effort: 节流总和惩罚（油门使用量）
        reward_effort = torch.clip(2.5 - self.effort, min=0).mean(dim=-1)  # [num_envs]
        reward_effort = reward_effort * self.reward_effort_weight
        
        # (b) throttle_smoothness: 节流变化惩罚（加速度平滑性）
        reward_throttle_smoothness = torch.clip(0.5 - self.drone.throttle_difference, min=0).mean(dim=-1)  # [num_envs]
        reward_throttle_smoothness = reward_throttle_smoothness * self.reward_throttle_smoothness_weight
        
        # (c) action_smoothness: 动作变化惩罚（动作一致性）
        action_diff = torch.norm((self.current_actions - self.prev_actions), dim=-1)  # [num_envs, n_drones]
        reward_action_smoothness = torch.clip(2.5 - action_diff, min=0).mean(dim=-1)  # [num_envs]
        reward_action_smoothness = reward_action_smoothness * self.reward_action_smoothness_weight
        
        # (d) spin: 偏航旋转惩罚（朝向稳定性）
        # drone_vel[..., -1] 是偏航角速度（yaw rate）
        y_spin = torch.abs(drone_vel[..., -1])  # [num_envs, n_drones] - yaw rate
        reward_spin = torch.clip(1.5 - y_spin, min=0).mean(dim=-1)  # [num_envs]
        reward_spin = reward_spin * self.reward_spin_weight
        
        # 2. Formation Maintenance Reward (using Laplacian distance from FormationUnified)
        
        # 2.1 Formation shape maintenance using Laplacian distance (更稳定、尺度不变)
        # Note: vmap is already imported at file level
        
        # 计算归一化和非归一化的 Laplacian 距离
        cost_l = vmap(cost_formation_laplacian)(
            drone_pos, 
            desired_L=self.formation_L,  # 归一化拉普拉斯矩阵
            normalized=True
        )
        
        cost_l_unnormalized = vmap(cost_formation_laplacian)(
            drone_pos, 
            desired_L=self.formation_L_unnormalized,  # 非归一化拉普拉斯矩阵
            normalized=False
        )
        
        # 为了兼容后续的 formation_cost 使用（用于终止条件），保留一个变量
        formation_cost = cost_l.squeeze(-1)
        
        # 2.2 编队形状奖励（主要）
        # [Modified] 移除魔法数字，改用标准的柯西核函数 (Cauchy Kernel)
        # 归一化误差：将 Laplacian 矩阵范数除以无人机数量，使其对编队规模不敏感
        normalized_formation_error = cost_l / self.drone.n
        
        # 形状容差参数 (sigma)：控制奖励函数的陡峭程度
        # 误差越小奖励越高，当误差等于 sigma 时奖励为 0.5。
        # 根据测试，0.05 能在保持鲁棒性的同时提供足够的梯度（0.16m偏差时奖励降至0.46）
        formation_sigma = 0.05 
        reward_formation = 1.0 / (1.0 + torch.square(normalized_formation_error / formation_sigma))
        
        # [Added] 弹性编队逻辑：计算距离感知因子
        # 计算编队中心到当前目标门的距离
        dist_to_gate = torch.norm(formation_center - gate_positions_current, dim=-1)
        
        # 定义弹性区域：距离门 2.0米以内开始变软，因子从 1.0 降至 0.2
        # 这样靠近门时，打破编队的惩罚权重降低
        formation_stiffness = torch.clamp(dist_to_gate / 2.0, 0.2, 1.0)
        
        # [Modified] 应用弹性因子到编队奖励权重
        reward_formation = reward_formation.squeeze(-1) * \
                           self.formation_reward_weight * \
                           formation_stiffness
        
        # 2.3 编队尺度奖励（辅助）- 确保编队大小符合预期
        pairwise_distances = torch.cdist(drone_pos, drone_pos)  # [num_envs, n_drones, n_drones]
        current_formation_size = pairwise_distances.max(dim=-2)[0].max(dim=-1)[0]  # [num_envs]
        
        size_delta = current_formation_size - self.standard_formation_size
        
        # [Modified] 移除魔法数字和冗余项，仅计算纯粹的尺度奖励
        # 尺度容差参数：允许 0.2 米左右的尺寸偏差 (编队直径约0.85m)
        size_sigma = 0.2
        reward_size = 1.0 / (1.0 + torch.square(size_delta / size_sigma))
        
        reward_size = reward_size * self.formation_size_reward_weight
        
        # 2.4 编队分离度奖励（避免过于靠近）- 参考 FormationUnified
        pairwise_distances_nondiag = pairwise_distances + torch.eye(self.drone.n, device=self.device) * 1e6  # 排除对角线
        min_distances = pairwise_distances_nondiag.min(dim=-1)[0].min(dim=-1)[0]  # [num_envs]
        
        # 使用软约束：距离越小惩罚越大
        reward_separation = torch.where(
            min_distances < self.safe_distance,
            -self.separation_penalty_weight * (self.safe_distance - min_distances),
            torch.zeros_like(min_distances)
        )
        
        # 合并编队奖励
        # reward_cohesion = reward_size + reward_separation
        
        # 3. Gate Traversal Rewards
        
        # 3.1 Gate approach reward (vectorized)
        # Check which environments have valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count  # [num_envs]
        
        # Get current gate positions using advanced indexing
        # (Re-using gate_positions_current calculated above for consistency)
        current_gate_pos = gate_positions_current
        gate_active = valid_gate_mask  # [num_envs]
        
        
        # 保持原有的编队中心距离计算，用于其他奖励
        gate_distance_center = torch.norm(formation_center - current_gate_pos, dim=-1)


        
        # ========================================
        # 3.3 逐个无人机检测门穿越并标记（在计算进度奖励之前）
        # ========================================
        traversed_envs_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        reward_gate_traversal = torch.zeros(self.num_envs, device=self.device)
        reward_bypass_gate_penalty = torch.zeros(self.num_envs, device=self.device)
        gate_bypass_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Check gate traversal for environments with valid gates
        valid_gate_mask = self.current_gate_idx < self.gate_count
        if valid_gate_mask.any():
            # Get relevant data for environments with valid gates
            valid_envs = torch.where(valid_gate_mask)[0]
            valid_gate_indices = self.current_gate_idx[valid_envs]  # [num_valid_envs]
            valid_gate_pos = self.gate_positions[valid_envs, valid_gate_indices]  # [num_valid_envs, 3]
            
            # 获取所有无人机的位置
            valid_drone_pos = drone_pos[valid_envs]  # [num_valid_envs, n_drones, 3]
            
            # 条件1：检查每个无人机的X坐标是否超过门的X坐标
            # 现在无人机是向X轴正方向飞行，所以"穿过门"意味着X坐标大于门的X坐标
            drone_passed_x = valid_drone_pos[:, :, 0] > valid_gate_pos[:, 0].unsqueeze(1)  # [num_valid_envs, n_drones]
            
            # 条件2：检查每个无人机是否距离门中心足够近
            dist_y = torch.abs(valid_drone_pos[..., 1] - valid_gate_pos.unsqueeze(1)[..., 1])
            dist_z = torch.abs(valid_drone_pos[..., 2] - valid_gate_pos.unsqueeze(1)[..., 2]) 
            
            # 使用宽松一点的阈值 (例如 0.8 倍尺寸作为核心区，或者 1.0 倍)
            # 必须同时满足宽度和高度限制
            is_in_width = dist_y < (self.gate_width / 2.0)
            is_in_height = dist_z < (self.gate_height / 2.0)
            
            drone_close_enough = is_in_width & is_in_height
            
            # 获取当前门对应的已穿越状态（统一获取一次，避免重复）
            current_gate_passed_status = self.drone_gate_passed[
                valid_envs, valid_gate_indices
            ]
            
            # ========================================
            # 新增：绕门惩罚 - 惩罚那些X轴超过门但没从门中心穿过的无人机
            # ========================================
            # 检测哪些无人机X轴超过门但距离门中心太远（绕门飞过）
            bypassed_gate = drone_passed_x & (~drone_close_enough)  # [num_valid_envs, n_drones]
            
            # 只惩罚那些之前没穿过门、现在绕过门的无人机
            newly_bypassed = bypassed_gate & (~current_gate_passed_status)  # [num_valid_envs, n_drones]
            
            # 统计每个环境中绕门的无人机数量
            num_bypassed = newly_bypassed.sum(dim=1)  # [num_valid_envs]
            
            # 计算惩罚：每个绕门的无人机给予大惩罚
            # 使用更大的惩罚力度，确保智能体明确知道这是错误的
            bypass_penalties = num_bypassed.float() * (-self.gate_progress_reward_weight * 0)  # 惩罚是穿越奖励的2倍
            
            # 分配惩罚
            reward_bypass_gate_penalty[valid_envs] = bypass_penalties
            
            # 设置绕门失败标志 - 如果有任何无人机绕门，则标记该环境失败
            # 这会在后续被用于终止episode
            gate_bypass_failure[valid_envs] = num_bypassed > 0
            
            # 将绕门的无人机也标记为"已处理"，避免后续再获得奖励或重复惩罚
            self.drone_gate_passed[valid_envs, valid_gate_indices] = (
                current_gate_passed_status | newly_bypassed
            )
            # ========================================
            
            # 每个无人机都需要同时满足两个条件才算通过：超过门 且 在门附近
            drone_traversed = drone_passed_x & drone_close_enough  # [num_valid_envs, n_drones]
            
            # 找出本帧新穿越的无人机（之前未穿越，现在穿越了）
            # 注意：这里使用更新后的状态，已经绕门的无人机不会再被计为正常穿越
            updated_gate_passed_status = self.drone_gate_passed[
                valid_envs, valid_gate_indices
            ]
            newly_traversed = drone_traversed & (~updated_gate_passed_status)  # [num_valid_envs, n_drones]
            
            # 统计每个环境中新穿越的无人机数量
            num_newly_traversed = newly_traversed.sum(dim=1)  # [num_valid_envs]
            
            # 计算奖励：每个新穿越的无人机获得固定奖励
            # 每个无人机穿越门的奖励 = gate_progress_reward_weight
            # 例如：gate_progress_reward_weight = 40，则每个无人机穿越获得40的奖励
            # 3个无人机都穿过门总共获得 3 × 40 = 120 的奖励
            traversal_rewards = num_newly_traversed.float() * self.gate_traversal_reward_weight
            
            # 分配奖励
            reward_gate_traversal[valid_envs] = traversal_rewards
            
            # 更新穿越状态：记录哪些无人机已经穿过了当前门（包括绕门和正常穿越）
            self.drone_gate_passed[valid_envs, valid_gate_indices] = (
                updated_gate_passed_status | newly_traversed
            )
            
            # 检查是否所有无人机都穿过了当前门
            all_drones_passed = self.drone_gate_passed[valid_envs, valid_gate_indices].all(dim=1)  # [num_valid_envs]
            
            if all_drones_passed.any():
                # 找出所有无人机都穿过门的环境
                fully_traversed_mask = torch.zeros(len(valid_envs), dtype=torch.bool, device=self.device)
                fully_traversed_mask[all_drones_passed] = True
                
                fully_traversed_envs = valid_envs[fully_traversed_mask]
                
                # 标记这些环境在本帧完成了门穿越（所有无人机都通过）
                traversed_envs_mask[fully_traversed_envs] = True
                
                # 更新门索引（只有当所有无人机都通过后才切换到下一个门）
                self.current_gate_idx[fully_traversed_envs] += 1  # 允许超过gate_count-1
                self.gates_passed[fully_traversed_envs] += 1
        
        # ========================================
        # 3.4 计算门进度奖励（所有阶段都计算，用于监控）
        # 参考 forest.py 的 reward_vel 实现：速度方向沿目标方向的投影
        # 【新逻辑】只有当无人机还未穿过当前门时，才计算门进度奖励
        # ========================================
        reward_gate_progress = torch.zeros(self.num_envs, device=self.device)
        
        # 所有阶段都计算门进度奖励（用于日志监控）
        # 只对"有有效门"且"本帧未穿越门"的环境计算进度奖励
        gate_progress_active = (self.current_gate_idx < self.gate_count) & (~traversed_envs_mask)

        # 标记每架无人机是否已经完成所有门的穿越
        drone_all_gates_passed = self.drone_gate_passed.all(dim=1)  # [num_envs, n_drones]
        
        if gate_progress_active.any():
            active_envs = torch.where(gate_progress_active)[0]
            
            # 使用当前的 current_gate_idx（对于本帧穿越的环境，已经指向下一个门）
            active_gate_indices = self.current_gate_idx[active_envs]
            current_gate_pos_active = self.gate_positions[active_envs, active_gate_indices]
            
            # 获取活跃环境的当前无人机位置和速度
            current_pos_active = drone_pos[active_envs]  # [active_envs, n_drones, 3]
            current_vel_active = drone_vel[active_envs, :, :3]  # [active_envs, n_drones, 3] 只取线速度
            
            # 计算每个无人机到门的方向向量
            to_gate_vec = current_gate_pos_active.unsqueeze(1) - current_pos_active  # [active_envs, n_drones, 3]
            distance_to_gate = to_gate_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [active_envs, n_drones, 1]
            gate_direction = to_gate_vec / distance_to_gate  # [active_envs, n_drones, 3] 归一化方向向量
            
            # 计算速度在门方向上的投影（点积）- 类似 forest.py 的 reward_vel
            # (drone_vel * gate_direction).sum(-1) 得到速度在目标方向的分量
            individual_vel_reward = (current_vel_active * gate_direction).sum(dim=-1)  # [active_envs, n_drones]
            
            # 【核心修改】对每个无人机分别判断：只有还未穿过当前门的无人机才计算门进度奖励
            # 获取当前门的穿越状态：[active_envs, n_drones]
            env_indices_active = torch.arange(len(active_envs), device=self.device)
            current_gate_passed_active = self.drone_gate_passed[active_envs, active_gate_indices[env_indices_active]]
            
            # 创建掩码：只有未穿过当前门的无人机才计算奖励
            # ~current_gate_passed_active: True表示该无人机还未穿过当前门
            active_drone_mask = ~current_gate_passed_active  # [active_envs, n_drones]
            individual_vel_reward = individual_vel_reward * active_drone_mask.float()  # [active_envs, n_drones]

            # 计算所有未穿过门的无人机的总速度奖励
            total_vel_reward = individual_vel_reward.sum(dim=-1)  # [active_envs]
            
            reward_gate_progress[active_envs] = total_vel_reward * self.gate_progress_reward_weight
        
        # ========================================
        # 4. Endpoint Progress Rewards
        # 参考 forest.py 的 reward_vel 实现：速度方向沿目标方向的投影
        # 【新逻辑】只有当无人机穿过门后，才开始计算终点进度奖励
        # ========================================
        
        # 4.1 Calculate individual drone distances to their target endpoints
        individual_endpoint_distances = torch.norm(drone_pos - self.end_positions, dim=-1)  # [num_envs, n_drones]

        # 4.2 Endpoint progress reward - 使用速度方向的投影奖励
        # 计算每个无人机到终点的方向向量
        to_endpoint_vec = self.end_positions - drone_pos  # [num_envs, n_drones, 3]
        distance_to_endpoint = to_endpoint_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)  # [num_envs, n_drones, 1]
        endpoint_direction = to_endpoint_vec / distance_to_endpoint  # [num_envs, n_drones, 3] 归一化方向向量
        
        # 获取无人机线速度（world frame）
        current_vel = drone_vel[:, :, :3]  # [num_envs, n_drones, 3]
        
        # 计算速度在终点方向上的投影（点积）- 类似 forest.py 的 reward_vel
        # (drone_vel * endpoint_direction).sum(-1) 得到速度在目标方向的分量
        individual_vel_to_endpoint = (current_vel * endpoint_direction).sum(dim=-1).clip(max=2.0)  # [num_envs, n_drones]
        
        # 【核心修改1】为每个无人机单独判断是否应该计算终点进度奖励
        # 只有当无人机穿过了当前门后，才开始计算终点进度奖励
        
        # 获取每个环境的当前门索引
        env_indices = torch.arange(self.num_envs, device=self.device)
        current_gate_idx_safe = self.current_gate_idx.clamp(0, self.gate_count - 1)
        
        # 获取每个无人机对当前门的穿越状态 [num_envs, n_drones]
        # True表示该无人机已经穿过当前门
        drone_passed_current_gate = self.drone_gate_passed[env_indices, current_gate_idx_safe]
        
        # 创建掩码：只有穿过当前门的无人机才计算终点进度奖励
        # drone_passed_current_gate: True表示已穿过，才计算终点奖励
        endpoint_reward_mask = drone_passed_current_gate.float()  # [num_envs, n_drones]
        
        # 应用掩码到速度奖励
        individual_vel_to_endpoint = individual_vel_to_endpoint * endpoint_reward_mask  # [num_envs, n_drones]
        
        # [Added] 检查是否所有门都已穿越
        # Stage 0 & 1: 只要飞向终点就给满分权重 (1.0)。
        #   - Stage 0: 学习基本飞行。
        #   - Stage 1: 学习编队。此时虽然不强制穿门，但由于"弹性编队因子"的存在，
        #     穿门在物理上阻力变小，智能体可能会探索出穿门路径，但不强求。
        # Stage 2: 开启抑制 (0.1)。
        #   - 此时如果不穿门，飞向终点的奖励只有 10%，迫使智能体必须完成穿门任务。
        
        if current_stage < 2:
            # Stage 0 and 1: 所有无人机都可以获得终点奖励（不需要穿门）
            # 移除endpoint_reward_mask的约束
            individual_vel_to_endpoint = (current_vel * endpoint_direction).sum(dim=-1).clip(max=2.0)  # 重新计算不带掩码的奖励
            endpoint_weight_factor = torch.ones(self.num_envs, device=self.device)
        else:
            # Stage 2: 应用门穿越约束
            # endpoint_reward_mask已经应用到individual_vel_to_endpoint
            # 同时检查是否所有门都已穿越
            all_gates_cleared = self.gates_passed >= self.gate_count
            endpoint_weight_factor = torch.where(all_gates_cleared, 1.0, 0.1)
        
        # 计算所有无人机的总速度奖励 (Modified: multiply by weight factor)
        reward_endpoint_progress = individual_vel_to_endpoint.sum(dim=-1) * \
                                   self.endpoint_progress_reward_weight * \
                                   endpoint_weight_factor
        
        # 更新距离记录（用于其他统计）
        self.last_endpoint_distances = individual_endpoint_distances
        
        # ========================================
        # 5. Safety and Collision Avoidance
        # ========================================
        
        # 5.1 Inter-drone collision penalty
        # NOTE: min_distances is already calculated in formation reward section (line ~1352)
        # with diagonal excluded, so we can reuse it here
        # min_distances = pairwise_distances_nondiag.min(dim=-1)[0].min(dim=-1)[0]  # Already computed above
        
        # 5.2 Collision penalty reward (applied when collision is detected)
        # This will be used in termination detection below
        reward_collision_penalty = torch.zeros(self.num_envs, device=self.device)

        
        # Critical collision
        collision_terminated = min_distances < 0.18
        
        # Apply collision penalty when collision is detected
        reward_collision_penalty[collision_terminated] = -self.collision_penalty_weight
        
        # ========================================
        # 6. Task Completion Bonuses
        # ========================================
        
        
        # ========================================
        # 8. TOTAL REWARD COMBINATION - CURRICULUM AWARE WITH SMOOTH TRANSITION
        # ========================================
        # 注意：所有奖励组件（endpoint_progress, gate_progress等）在所有阶段都会计算
        # 这样可以在 wandb 中监控所有指标，但只有特定阶段的奖励会加入总 return
        
        # 定义三个阶段的奖励组合
        reward_stage_0 = (
            # Stage 0: Endpoint only - focus on basic flight and endpoint reaching
            # Basic flight rewards
            reward_uprightness +
            reward_heading +
            # Smoothness rewards
            reward_throttle_smoothness + reward_action_smoothness + reward_spin +
            # Endpoint progress (heavily weighted)
            reward_endpoint_progress
            # 注意：不加入 formation 和 gate 相关奖励
        )
        
        reward_stage_1 = (
            # Stage 1: Formation + Endpoint - no gates
            # Basic flight rewards
            reward_uprightness +
            reward_heading +
            # Smoothness rewards
            reward_throttle_smoothness + reward_action_smoothness + reward_spin +
            # Formation rewards (important)
            reward_formation + reward_size + reward_separation +
            # Endpoint progress
            reward_endpoint_progress
            # 注意：不加入 gate 相关奖励
        )
        
        reward_stage_2 = (
            # Stage 2: Full task - Formation + Gates + Endpoint
            # Basic flight rewards
            reward_uprightness +
            reward_heading +
            # Smoothness rewards
            reward_throttle_smoothness + reward_action_smoothness + reward_spin +
            # Formation rewards
            reward_formation + reward_size + reward_separation +
            # Gate traversal rewards
            reward_gate_progress + reward_gate_traversal + reward_bypass_gate_penalty +
            # Endpoint progress rewards
            reward_endpoint_progress
        )
        
        # 根据当前阶段和过渡权重计算最终奖励
        if self.enable_smooth_transition and self.transition_start_frame >= 0:
            # 在过渡期内，使用线性插值混合奖励
            transition_weight = self.get_transition_weight()  # Get current transition weight
            
            # 根据过渡的起始和目标阶段进行插值
            if self.transition_from_stage == 0 and self.transition_to_stage == 1:
                # Stage 0 -> Stage 1 过渡
                total_reward = (1.0 - transition_weight) * reward_stage_0 + transition_weight * reward_stage_1
            elif self.transition_from_stage == 1 and self.transition_to_stage == 2:
                # Stage 1 -> Stage 2 过渡
                total_reward = (1.0 - transition_weight) * reward_stage_1 + transition_weight * reward_stage_2
            else:
                # 其他情况（不应该发生），使用当前阶段奖励
                if current_stage == 0:
                    total_reward = reward_stage_0
                elif current_stage == 1:
                    total_reward = reward_stage_1
                else:
                    total_reward = reward_stage_2
        else:
            # 不在过渡期，直接使用当前阶段的奖励
            if current_stage == 0:
                total_reward = reward_stage_0
            elif current_stage == 1:
                total_reward = reward_stage_1
            else:
                total_reward = reward_stage_2
        
        # Expand reward to all drones (shared reward)
        reward = total_reward.unsqueeze(-1).unsqueeze(-1).expand(-1, self.drone.n, 1)
        
        # ====================
        # TERMINATION CONDITIONS
        # ====================
        
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        
        # Formation breakdown - more lenient threshold for direct learning
        # Use formation_cost as a measure of formation breakdown
        formation_breakdown = formation_cost > (self.formation_tolerance * 2.0)
        
        # Task success - all gates passed
        success = self.gates_passed >= self.gate_count
        
        # Out of bounds
        max_height = 4.0
        min_height = 0.1
        out_of_bounds = (drone_pos[..., 2] > max_height) | (drone_pos[..., 2] < min_height)
        out_of_bounds = out_of_bounds.any(dim=-1)
        
        # 新增终止条件2: 无人机超过终点过远就终止 - 合理版本
        endpoint_exceeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # # 使用正确的终点位置计算
        # endpoint_positions = self.end_positions[:, 0, :]  # [num_envs, 3] 取第一个无人机的终点作为参考

        # # 方法1: 使用编队中心的终点
        # endpoint_positions = self.end_positions.mean(dim=1)  # [num_envs, 3]

        # 方法2: 计算每个无人机到各自终点的距离
        endpoint_threshold = 0.5  # 终点距离阈值（米）
        individual_endpoint_distances = torch.norm(drone_pos - self.end_positions, dim=-1)
        any_drone_too_far = (individual_endpoint_distances > endpoint_threshold).any(dim=1)
    
        
        # 同时检查无人机是否真的向终点方向移动了（避免在起始位置就判断为超出）
        # 现在无人机向X轴正方向飞行，检查无人机是否超过了终点的X位置（即X坐标大于终点X坐标）
        drone_x_positions = drone_pos[:, :, 0]  # [num_envs, num_drones]
        endpoint_x_positions = self.end_positions[:,:, 0]  # [num_envs, num_drones]
        any_drone_passed_endpoint_x = (drone_x_positions > endpoint_x_positions + 0.5).any(dim=1)  # [num_envs] 0.5米缓冲

        # 只有当无人机既超过了终点X位置又距离终点过远时才终止
        endpoint_exceeded = any_drone_too_far & any_drone_passed_endpoint_x

        # ===== 课程学习：终止条件根据阶段调整 =====
        # 基础终止条件（所有阶段）：碰撞、出界、终点超出
        terminated = collision_terminated | out_of_bounds | endpoint_exceeded
        
        # Eval mode specific termination: Stop if all drones reach the endpoint
        if self.eval:
            reached_threshold = 0.05 # Distance threshold in meters
            all_drones_reached = (individual_endpoint_distances < reached_threshold).all(dim=1)
            terminated = terminated | all_drones_reached

        # 第三阶段（Stage 2）才启用绕门失败终止
        # [Modified] 移除绕门终止条件，保留惩罚
        if current_stage == 2:
            pass
            # terminated = terminated | gate_bypass_failure

        # if terminated.any():
        #     print("collision_terminated:", collision_terminated)
        #     print("gate_bypass_failure:", gate_bypass_failure)
        #     print("success:", success)
        #     print("out_of_bounds:", out_of_bounds)
        #     print("endpoint_exceeded:", endpoint_exceeded)
        

        # ==================== STATS UPDATE ====================
        truncated = self.progress_buf >= self.max_episode_length
        
        # Basic episode metrics
        self.stats["return"] += total_reward.unsqueeze(-1).expand(-1, self.drone.n)

        
        # Reward component tracking (current step values)
        self.stats["reward_uprightness"] += reward_uprightness.unsqueeze(-1)
        self.stats["reward_heading"] += reward_heading.unsqueeze(-1)
        
        # Smoothness reward components (matching FormationUnified)
        self.stats["reward_effort"] += reward_effort.unsqueeze(-1)
        self.stats["reward_throttle_smoothness"] += reward_throttle_smoothness.unsqueeze(-1)
        self.stats["reward_action_smoothness"] += reward_action_smoothness.unsqueeze(-1)
        self.stats["reward_spin"] += reward_spin.unsqueeze(-1)
        
        # Formation reward components
        self.stats["reward_formation"] += reward_formation.unsqueeze(-1)
        self.stats["reward_size"] += reward_size.unsqueeze(-1)
        self.stats["reward_separation"] += reward_separation.unsqueeze(-1)
        
        # Progress rewards
        self.stats["reward_endpoint_progress"] += reward_endpoint_progress.unsqueeze(-1)
        self.stats["reward_gate_progress"] += reward_gate_progress.unsqueeze(-1)
        self.stats["reward_gate_traversal"] += reward_gate_traversal.unsqueeze(-1)
        self.stats["reward_bypass_gate_penalty"] += reward_bypass_gate_penalty.unsqueeze(-1)
        
        # Safety reward components
        self.stats["reward_collision_penalty"] += reward_collision_penalty.unsqueeze(-1)
        

        # Update previous state for next iteration
        self.prev_drone_velocities = drone_vel[..., :3].clone()
        self.prev_drone_positions = drone_pos.clone()
        if hasattr(self, 'actions'):
            self.prev_actions = self.actions.clone()

        terminated = terminated.unsqueeze(-1)
        truncated = (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        
        if self.eval:
            self._log_eval_data(success, terminated, truncated)

        return TensorDict(
            {
                "agents": {"reward": reward},
                "done": terminated | truncated,
                "terminated": terminated,
                "truncated": truncated,
            },
            self.batch_size,
        )
    def _update_trajectory_visualization(self):
        """Update drone trajectory visualization with start points, end points, flight paths, and formation connections."""
        if not self.visualization_enabled or not self._should_render(0):
            return
            
        # Clear previous visualization
        self.draw.clear_lines()
        
        # Get current drone positions
        drone_pos, _ = self.drone.get_world_poses()
        
        # Update trajectory history for each environment
        for env_idx in range(self.num_envs):
            step_idx = int(self.trajectory_step_count[env_idx] % self.trajectory_max_length)
            self.drone_trajectory_history[env_idx, :, step_idx] = drone_pos[env_idx]
            
            # Update formation history (sample every N steps)
            if self.trajectory_step_count[env_idx] % self.formation_sample_interval == 0:
                sample_idx = int(self.formation_sample_count[env_idx] % self.formation_max_samples)
                self.formation_history[env_idx, sample_idx] = drone_pos[env_idx]
                self.formation_sample_count[env_idx] += 1
            
        self.trajectory_step_count += 1
        
        # Visualize only the central environment for clarity
        # Use the environment closest to the origin (defaultGroundPlane center)
        central_env_idx = self.central_env_idx
        
        # Draw start positions (green spheres)
        start_pos_list = self.start_positions[central_env_idx].cpu().tolist()
        for i, pos in enumerate(start_pos_list):
            # Draw start point as a small cross
            offset = 0.1
            cross_points_0 = [
                [pos[0] - offset, pos[1], pos[2]],
                [pos[0], pos[1] - offset, pos[2]],
                [pos[0], pos[1], pos[2] - offset]
            ]
            cross_points_1 = [
                [pos[0] + offset, pos[1], pos[2]],
                [pos[0], pos[1] + offset, pos[2]],
                [pos[0], pos[1], pos[2] + offset]
            ]
            
            # Green color for start positions
            colors = [(0, 1, 0, 1)] * len(cross_points_0)  # Green
            sizes = [2.0] * len(cross_points_0)
            
            self.draw.draw_lines(cross_points_0, cross_points_1, colors, sizes)
        
        # Draw end positions (red spheres)
        end_pos_list = self.end_positions[central_env_idx].cpu().tolist()
        for i, pos in enumerate(end_pos_list):
            # Draw end point as a small cross
            offset = 0.1
            cross_points_0 = [
                [pos[0] - offset, pos[1], pos[2]],
                [pos[0], pos[1] - offset, pos[2]],
                [pos[0], pos[1], pos[2] - offset]
            ]
            cross_points_1 = [
                [pos[0] + offset, pos[1], pos[2]],
                [pos[0], pos[1] + offset, pos[2]],
                [pos[0], pos[1], pos[2] + offset]
            ]
            
            # Red color for end positions
            colors = [(1, 0, 0, 1)] * len(cross_points_0)  # Red
            sizes = [2.0] * len(cross_points_0)
            
            self.draw.draw_lines(cross_points_0, cross_points_1, colors, sizes)
        
        # Draw gate positions (yellow frames) - corrected to show rotation
        # Get gate positions and rotations for central environment
        gate_pos_tensor = self.gate_positions[central_env_idx]  # [gate_count, 3]
        gate_rot_tensor = self.gate_rotations[central_env_idx]  # [gate_count, 4]
        
        # Use _compute_gate_corners to calculate rotated corners
        # Returns [gate_count, 4, 3] where 4 corners are: [bottom-left, top-left, bottom-right, top-right]
        gate_corners_tensor = self._compute_gate_corners(gate_pos_tensor, gate_rot_tensor)
        
        # Convert to list for drawing
        gate_corners_list = gate_corners_tensor.cpu().tolist()
        
        for gate_idx, corners in enumerate(gate_corners_list):
            # corners is a list of 4 points: [bottom-left, top-left, bottom-right, top-right]
            # Draw rectangle frame by connecting: bottom-left -> top-left -> top-right -> bottom-right -> bottom-left
            
            # Connect corners in order to form a rectangle
            frame_lines_0 = [
                corners[0],  # bottom-left -> top-left
                corners[1],  # top-left -> top-right
                corners[3],  # top-right -> bottom-right
                corners[2]   # bottom-right -> bottom-left
            ]
            frame_lines_1 = [
                corners[1],  # top-left
                corners[3],  # top-right
                corners[2],  # bottom-right
                corners[0]   # bottom-left
            ]
            
            # Yellow color for gates
            colors = [(1, 1, 0, 1)] * len(frame_lines_0)  # Yellow
            sizes = [3.0] * len(frame_lines_0)
            
            self.draw.draw_lines(frame_lines_0, frame_lines_1, colors, sizes)
            
            # Optional: Draw pendulum arm from pivot to gate center
            # Get pivot point for this gate
            pivot = self.gate_pivot_points[central_env_idx, gate_idx].cpu().tolist()
            gate_center = gate_pos_tensor[gate_idx].cpu().tolist()
            
            # Draw arm as a thin gray line
            arm_lines_0 = [pivot]
            arm_lines_1 = [gate_center]
            arm_colors = [(0.5, 0.5, 0.5, 0.6)]  # Gray, semi-transparent
            arm_sizes = [1.5]
            
            self.draw.draw_lines(arm_lines_0, arm_lines_1, arm_colors, arm_sizes)
        
        # Draw drone trajectories (blue lines)
        current_step = int(self.trajectory_step_count[central_env_idx].item())
        if current_step > 1:
            trajectory_length = min(current_step, self.trajectory_max_length)
            
            for drone_idx in range(self.drone.n):
                # Get trajectory points for this drone
                drone_trajectory = self.drone_trajectory_history[central_env_idx, drone_idx]
                
                # Extract valid trajectory points
                valid_points = []
                for step in range(trajectory_length - 1):
                    point_idx = (current_step - trajectory_length + step) % self.trajectory_max_length
                    pos = drone_trajectory[point_idx].cpu().tolist()
                    valid_points.append(pos)
                
                if len(valid_points) > 1:
                    # Create line segments
                    point_list_0 = valid_points[:-1]
                    point_list_1 = valid_points[1:]
                    
                    # Different color for each drone
                    color = [0.2 + 0.3 * drone_idx, 0.5, 1.0, 0.8]  # Blue gradient
                    colors = [color] * len(point_list_0)
                    sizes = [1.5] * len(point_list_0)
                    
                    self.draw.draw_lines(point_list_0, point_list_1, colors, sizes)
        
        # Draw formation connections (historical snapshots)
        current_formation_samples = int(self.formation_sample_count[central_env_idx].item())
        if current_formation_samples > 0:
            # Calculate how many samples to show
            max_samples_to_show = min(current_formation_samples, self.formation_max_samples)
            
            for sample_idx in range(max_samples_to_show):
                # Calculate the actual index in the circular buffer
                actual_idx = (current_formation_samples - max_samples_to_show + sample_idx) % self.formation_max_samples
                formation_snapshot = self.formation_history[central_env_idx, actual_idx]  # [num_drones, 3]
                
                # Create all possible connections between drones (complete graph)
                formation_points = formation_snapshot.cpu().tolist()
                
                connection_lines_0 = []
                connection_lines_1 = []
                
                for i in range(len(formation_points)):
                    for j in range(i + 1, len(formation_points)):
                        connection_lines_0.append(formation_points[i])
                        connection_lines_1.append(formation_points[j])
                
                if len(connection_lines_0) > 0:
                    # Use different alpha for different time samples (older = more transparent)
                    age_factor = sample_idx / max(max_samples_to_show - 1, 1)  # 0 (oldest) to 1 (newest)
                    alpha = 0.2 + 0.6 * age_factor  # Alpha from 0.2 (oldest) to 0.8 (newest)
                    
                    # Use magenta/purple color for formation connections
                    formation_color = (1.0, 0.2, 1.0, alpha)  # Magenta with varying alpha
                    colors = [formation_color] * len(connection_lines_0)
                    
                    # Thinner lines for formation connections
                    sizes = [0.8] * len(connection_lines_0)
                    
                    self.draw.draw_lines(connection_lines_0, connection_lines_1, colors, sizes)
        
        # Draw current formation connections (real-time, bright color)
        current_drone_positions = drone_pos[central_env_idx].cpu().tolist()  # [num_drones, 3]
        
        current_connection_lines_0 = []
        current_connection_lines_1 = []
        
        for i in range(len(current_drone_positions)):
            for j in range(i + 1, len(current_drone_positions)):
                current_connection_lines_0.append(current_drone_positions[i])
                current_connection_lines_1.append(current_drone_positions[j])
        
        if len(current_connection_lines_0) > 0:
            # Bright cyan color for current formation
            current_formation_color = (0.0, 1.0, 1.0, 1.0)  # Bright cyan, full opacity
            colors = [current_formation_color] * len(current_connection_lines_0)
            sizes = [1.2] * len(current_connection_lines_0)  # Slightly thicker for current formation
            
            self.draw.draw_lines(current_connection_lines_0, current_connection_lines_1, colors, sizes)

    def update_curriculum_stage(self, current_frames: int, total_frames: int):
        """Update curriculum learning stage based on training progress.
        
        Args:
            current_frames: Current number of frames trained
            total_frames: Total training frames configured
            
        Returns:
            stage_changed: Whether the stage has changed
        """
        if not self.curriculum_enable:
            return False
        
        old_stage = self.current_stage
        self.total_frames = total_frames
        self.current_global_frames = current_frames  # Update global frame count
        
        # Calculate stage boundaries based on total configured frames
        stage1_end = int(self.stage_ratios[0] * total_frames)
        stage2_end = int((self.stage_ratios[0] + self.stage_ratios[1]) * total_frames)
        
        if current_frames < stage1_end:
            self.current_stage = 0  # Stage 1: Endpoint only
        elif current_frames < stage2_end:
            self.current_stage = 1  # Stage 2: Formation
        else:
            self.current_stage = 2  # Stage 3: Formation + Gates
        
        stage_changed = (old_stage != self.current_stage)
        if stage_changed:
            print(f"[Curriculum] Stage changed from {old_stage} to {self.current_stage} at {current_frames}/{total_frames} frames")
            print(f"[Curriculum] Stage boundaries: 0-{stage1_end} (endpoint), {stage1_end}-{stage2_end} (formation), {stage2_end}+ (full task)")
            
            # Start smooth transition if enabled
            if self.enable_smooth_transition:
                self.transition_start_frame = current_frames
                self.transition_from_stage = old_stage
                self.transition_to_stage = self.current_stage
                print(f"[Curriculum] Starting smooth transition from stage {old_stage} to {self.current_stage}")
                print(f"[Curriculum] Transition will last for {self.transition_frames} frames")
        
        return stage_changed
    
    def get_curriculum_stage(self) -> int:
        """Get current curriculum learning stage.
        
        Returns:
            current_stage: 0 (endpoint), 1 (formation), 2 (formation+gates)
        """
        return self.current_stage if self.curriculum_enable else 2
    
    def get_transition_weight(self) -> float:
        """Calculate transition weight for smooth reward blending.
        
        Returns:
            weight: Weight for new stage reward (0.0 to 1.0)
                   0.0 = 100% old stage, 1.0 = 100% new stage
        """
        if not self.enable_smooth_transition or self.transition_start_frame < 0:
            return 1.0  # Not in transition, use full new stage reward
        
        frames_since_transition = self.current_global_frames - self.transition_start_frame
        
        if frames_since_transition >= self.transition_frames:
            # Transition完成，清除过渡状态
            self.transition_start_frame = -1
            self.transition_from_stage = -1
            self.transition_to_stage = -1
            return 1.0
        
        # 线性插值：从 0.0 (开始) 到 1.0 (结束)
        weight = frames_since_transition / self.transition_frames
        return weight


    def _log_eval_data(self, success, terminated, truncated):
        if not self.eval:
            return
            
        # Get current state
        pos, rot = self.drone.get_world_poses()
        vel = self.drone.get_velocities()
        
        # Collect data for this step
        step_data = {
            "drone_pos": pos.clone().cpu(),
            "drone_rot": rot.clone().cpu(),
            "drone_vel": vel[..., :3].clone().cpu(),
            "drone_ang_vel": vel[..., 3:].clone().cpu(),
            "gate_pos": self.gate_positions.clone().cpu(),
            "gate_rot": self.gate_rotations.clone().cpu(),
            "success": success.clone().cpu(),
            "terminated": terminated.clone().cpu(),
            "truncated": truncated.clone().cpu(),
            "env_frames": self.progress_buf.clone().cpu(),
            "current_gate_idx": self.current_gate_idx.clone().cpu(),
            "drone_gate_passed": self.drone_gate_passed.clone().cpu()
        }
        self.eval_data_log.append(step_data)

    def pop_eval_data(self):
        """Retrieve and clear the current evaluation data buffer."""
        if not self.eval_data_log:
            return None
        
        # Stack data: List[Dict] -> Dict[Tensor]
        keys = self.eval_data_log[0].keys()
        stacked_data = {}
        for k in keys:
            # Stack along time dimension (dim=0)
            stacked_data[k] = torch.stack([d[k] for d in self.eval_data_log], dim=0)
        
        self.eval_data_log = [] # Clear buffer after retrieval
        return stacked_data


def cost_formation_hausdorff(p: torch.Tensor, desired_p: torch.Tensor) -> torch.Tensor:
    """
    Calculate Hausdorff distance-based formation cost.
    
    Args:
        p: Current positions [batch_size, n_drones, 3]
        desired_p: Desired formation positions [batch_size, n_drones, 3] or [n_drones, 3]
        
    Returns:
        cost: Formation cost [batch_size, 1]
    """
    # Center both formations at their centroids
    p_centered = p - p.mean(-2, keepdim=True)
    if desired_p.dim() == 2:
        # If desired_p is 2D, expand to match batch size
        desired_p = desired_p.unsqueeze(0).expand(p.shape[0], -1, -1)
    desired_p_centered = desired_p - desired_p.mean(-2, keepdim=True)
    
    # Calculate bidirectional Hausdorff distance
    cost = torch.max(
        directed_hausdorff(p_centered, desired_p_centered), 
        directed_hausdorff(desired_p_centered, p_centered)
    )
    return cost.unsqueeze(-1)


def directed_hausdorff(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Calculate directed Hausdorff distance.
    
    Args:
        p: Source points [batch_size, n, 3]
        q: Target points [batch_size, m, 3]
        
    Returns:
        distance: Directed Hausdorff distance [batch_size]
    """
    d = torch.cdist(p, q, p=2).min(-1).values.max(-1).values
    return d


def cost_formation_laplacian(
    p: torch.Tensor,
    desired_L: torch.Tensor,
    normalized=False,
) -> torch.Tensor:
    """
    A scale and translation invariant formation similarity cost
    
    Args:
        p: Current positions [n_drones, 3]
        desired_L: Desired Laplacian matrix [n_drones, n_drones]
        normalized: Whether to use normalized Laplacian
        
    Returns:
        cost: Formation cost [1]
    """
    L = laplacian(p, normalized)
    cost = torch.linalg.matrix_norm(desired_L - L)
    return cost.unsqueeze(-1)


def laplacian(p: torch.Tensor, normalize=False):
    """
    Compute symmetric normalized laplacian
    
    Args:
        p: Positions [n, dim]
        normalize: Whether to normalize
        
    Returns:
        L: Laplacian matrix [n, n]
    """
    assert p.dim() == 2
    A = torch.cdist(p, p)  # A[i, j] = norm_2(p[i], p[j]), A.shape = [n, n]
    D = torch.sum(A, dim=-1)  # D[i] = \sum_{j=1}^n norm_2(p[i], p[j]), D.shape = [n, ]
    if normalize:
        DD = D**-0.5
        A = torch.einsum("i,ij->ij", DD, A)
        A = torch.einsum("ij,j->ij", A, DD)
        L = torch.eye(p.shape[0], device=p.device) - A
    else:
        L = torch.diag(D) - A
    return L
