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


import copy
import logging
from typing import Type, Dict

import torch
import torch.distributions as D
import yaml
from torch.func import vmap
from tensordict.nn import make_functional
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from tensordict import TensorDict

from omni_drones.views import RigidPrimView
from omni_drones.actuators.rotor_group import RotorGroup
from omni_drones.controllers import ControllerBase
from omni_drones.robots import RobotBase, RobotCfg
from omni_drones.robots.drone.dynamics import QuadrotorDynamicsRK4
from omni_drones.utils.torch import (
    normalize, off_diag, quat_rotate, quat_rotate_inverse, quat_axis, symlog
)

from dataclasses import dataclass, is_dataclass
from collections import defaultdict

import pprint


class MultirotorBase(RobotBase):

    param_path: str

    def __init__(
        self,
        name: str = None,
        cfg: RobotCfg=None,
        is_articulation: bool = True,
    ) -> None:
        super().__init__(name, cfg, is_articulation)

        with open(self.param_path, "r") as f:
            logging.info(f"Reading {self.name}'s params from {self.param_path}.")
            self.params = yaml.safe_load(f)
        self.num_rotors = self.params["rotor_configuration"]["num_rotors"]

        self.intrinsics_spec = CompositeSpec({
            "mass": UnboundedContinuousTensorSpec(1),
            "inertia": UnboundedContinuousTensorSpec(3),
            "com": UnboundedContinuousTensorSpec(3),
            "KF": UnboundedContinuousTensorSpec(self.num_rotors),
            "KM": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_up": UnboundedContinuousTensorSpec(self.num_rotors),
            "tau_down": UnboundedContinuousTensorSpec(self.num_rotors),
            "drag_coef": UnboundedContinuousTensorSpec(1),
        }).to(self.device)

        state_dim = 19 + self.num_rotors
        self.state_spec = UnboundedContinuousTensorSpec(state_dim, device=self.device)
        self.randomization = defaultdict(dict)
        
        # 读取仿真模式配置
        # sim_mode 可以是 "physics" (使用 Isaac Sim 物理引擎) 或 "kinematic" (使用数学模型)
        self.sim_mode = "physics"  # 默认使用物理引擎
        if cfg is not None:
            if isinstance(cfg, dict):
                self.sim_mode = cfg.get("sim_mode", "physics")
            elif hasattr(cfg, "sim_mode"):
                self.sim_mode = cfg.sim_mode
        
        logging.info(f"[MultirotorBase] Drone '{self.name}' initialized with sim_mode: {self.sim_mode}")
        
        # 如果使用运动学模式，初始化动力学模型
        if self.sim_mode == "kinematic":
            # 质量参数将在 initialize() 后设置
            # 这里先占位，后面会更新
            self.dynamics_model = None
            self._dynamics_initialized = False

    @property
    def action_spec(self):
        if not hasattr(self, "_action_spec"):
            self._action_spec = BoundedTensorSpec(-1, 1, self.num_rotors, device=self.device)
        return self._action_spec

    def initialize(
        self,
        prim_paths_expr: str = None,
        track_contact_forces: bool = False
    ):
        if self.is_articulation:
            super().initialize(prim_paths_expr=prim_paths_expr)
            self.base_link = RigidPrimView(
                prim_paths_expr=f"{self.prim_paths_expr}/base_link",
                name="base_link",
                track_contact_forces=track_contact_forces,
                shape=self.shape,
            )
            self.base_link.initialize()
            print(self._view.dof_names)
            print(self._view._dof_indices)
            rotor_joint_indices = [
                i for i, dof_name in enumerate(self._view._dof_names)
                if dof_name.startswith("rotor")
            ]
            if len(rotor_joint_indices):
                self.rotor_joint_indices = torch.tensor(
                    rotor_joint_indices,
                    device=self.device
                )
            else:
                self.rotor_joint_indices = None
        else:
            super().initialize(prim_paths_expr=f"{prim_paths_expr}/base_link")
            self.base_link = self._view
            self.prim_paths_expr = prim_paths_expr

        self.rotors_view = RigidPrimView(
            # prim_paths_expr=f"{self.prim_paths_expr}/rotor_[0-{self.num_rotors-1}]",
            prim_paths_expr=f"{self.prim_paths_expr}/rotor_*",
            name="rotors",
            shape=(*self.shape, self.num_rotors)
        )
        self.rotors_view.initialize()

        rotor_config = self.params["rotor_configuration"]
        self.rotors = RotorGroup(rotor_config, dt=self.dt).to(self.device)

        rotor_params = make_functional(self.rotors)
        self.KF_0 = rotor_params["KF"].clone()
        self.KM_0 = rotor_params["KM"].clone()
        self.MAX_ROT_VEL = (
            torch.as_tensor(rotor_config["max_rotation_velocities"])
            .float()
            .to(self.device)
        )
        self.rotor_params = rotor_params.expand(self.shape).clone()

        self.tau_up = self.rotor_params["tau_up"]
        self.tau_down = self.rotor_params["tau_down"]
        self.KF = self.rotor_params["KF"]
        self.KM = self.rotor_params["KM"]
        self.throttle = self.rotor_params["throttle"]
        self.directions = self.rotor_params["directions"]

        self.thrusts = torch.zeros(*self.shape, self.num_rotors, 3, device=self.device)
        self.torques = torch.zeros(*self.shape, 3, device=self.device)
        self.forces = torch.zeros(*self.shape, 3, device=self.device)

        self.pos, self.rot = self.get_world_poses(True)
        self.throttle_difference = torch.zeros(self.throttle.shape[:-1], device=self.device)
        self.heading = torch.zeros(*self.shape, 3, device=self.device)
        self.up = torch.zeros(*self.shape, 3, device=self.device)
        self.vel = self.vel_w = torch.zeros(*self.shape, 6, device=self.device)
        self.vel_b = torch.zeros_like(self.vel_w)
        self.acc = self.acc_w = torch.zeros(*self.shape, 6, device=self.device)
        self.acc_b = torch.zeros_like(self.acc_w)

        # self.jerk = torch.zeros(*self.shape, 6, device=self.device)
        self.alpha = 0.9

        self.masses = self.base_link.get_masses().clone()
        self.gravity = self.masses * 9.81
        self.inertias = self.base_link.get_inertias().reshape(*self.shape, 3, 3).diagonal(0, -2, -1)
        # default/initial parameters
        self.MASS_0 = self.masses[0].clone()
        self.INERTIA_0 = (
            self.base_link
            .get_inertias()
            .reshape(*self.shape, 3, 3)[0]
            .diagonal(0, -2, -1)
            .clone()
        )
        self.THRUST2WEIGHT_0 = self.KF_0 / (self.MASS_0 * 9.81) # TODO: get the real g
        self.FORCE2MOMENT_0 = torch.broadcast_to(self.KF_0 / self.KM_0, self.THRUST2WEIGHT_0.shape)

        logging.info(str(self))

        self.drag_coef = torch.zeros(*self.shape, 1, device=self.device) * self.params["drag_coef"]
        self.intrinsics = self.intrinsics_spec.expand(self.shape).zero()
        
        # 如果使用运动学模式，初始化动力学模型
        if self.sim_mode == "kinematic" and not self._dynamics_initialized:
            # 获取平均质量（假设所有无人机质量相同或接近）
            avg_mass = self.masses.mean().item()
            logging.info(f"[MultirotorBase] Initializing kinematic dynamics model with mass={avg_mass:.3f}kg, dt={self.dt:.4f}s")
            self.dynamics_model = QuadrotorDynamicsRK4(mass=avg_mass, g=9.81, dt=self.dt).to(self.device)
            self._dynamics_initialized = True

    def setup_randomization(self, cfg):
        if not self.initialized:
            raise RuntimeError

        for phase in ("train", "eval"):
            if phase not in cfg: continue
            mass_scale = cfg[phase].get("mass_scale", None)
            if mass_scale is not None:
                low = self.MASS_0 * mass_scale[0]
                high = self.MASS_0 * mass_scale[1]
                self.randomization[phase]["mass"] = D.Uniform(low, high)
            inertia_scale = cfg[phase].get("inertia_scale", None)
            if inertia_scale is not None:
                low = self.INERTIA_0 * torch.as_tensor(inertia_scale[0], device=self.device)
                high = self.INERTIA_0 * torch.as_tensor(inertia_scale[1], device=self.device)
                self.randomization[phase]["inertia"] = D.Uniform(low, high)
            t2w_scale = cfg[phase].get("t2w_scale", None)
            if t2w_scale is not None:
                low = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[0], device=self.device)
                high = self.THRUST2WEIGHT_0 * torch.as_tensor(t2w_scale[1], device=self.device)
                self.randomization[phase]["thrust2weight"] = D.Uniform(low, high)
            f2m_scale = cfg[phase].get("f2m_scale", None)
            if f2m_scale is not None:
                low = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[0], device=self.device)
                high = self.FORCE2MOMENT_0 * torch.as_tensor(f2m_scale[1], device=self.device)
                self.randomization[phase]["force2moment"] = D.Uniform(low, high)
            drag_coef_scale = cfg[phase].get("drag_coef_scale", None)
            if drag_coef_scale is not None:
                low = self.params["drag_coef"] * drag_coef_scale[0]
                high = self.params["drag_coef"] * drag_coef_scale[1]
                self.randomization[phase]["drag_coef"] = D.Uniform(
                    torch.tensor(low, device=self.device),
                    torch.tensor(high, device=self.device)
                )
            tau_up = cfg[phase].get("tau_up", None)
            if tau_up is not None:
                self.randomization[phase]["tau_up"] = D.Uniform(
                    torch.tensor(tau_up[0], device=self.device),
                    torch.tensor(tau_up[1], device=self.device)
                )
            tau_down = cfg[phase].get("tau_down", None)
            if tau_down is not None:
                self.randomization[phase]["tau_down"] = D.Uniform(
                    torch.tensor(tau_down[0], device=self.device),
                    torch.tensor(tau_down[1], device=self.device)
                )
            com = cfg[phase].get("com", None)
            if com is not None:
                self.randomization[phase]["com"] = D.Uniform(
                    torch.tensor(com[0], device=self.device),
                    torch.tensor(com[1], device=self.device)
                )
            if not len(self.randomization[phase]) == len(cfg[phase]):
                unkown_keys = set(cfg[phase].keys()) - set(self.randomization[phase].keys())
                raise ValueError(
                    f"Unknown randomization {unkown_keys}."
                )

        logging.info(f"Setup randomization:\n" + pprint.pformat(dict(self.randomization)))

    def apply_action(self, actions: torch.Tensor, target_rate: torch.Tensor = None, target_thrust: torch.Tensor = None) -> torch.Tensor:
        """
        应用动作到无人机
        
        根据 sim_mode 选择不同的实现:
        - physics: 使用 Isaac Sim 物理引擎（需要 actions = rotor_commands）
        - kinematic: 使用内部动力学模型（需要 target_rate + target_thrust）
        
        Args:
            actions: [batch, n_drones, num_rotors] 电机转速指令 (归一化 -1到1)
                     仅 physics 模式使用
            target_rate: [batch, n_drones, 3] 目标角速度 (rad/s)
                         仅 kinematic 模式使用
            target_thrust: [batch, n_drones, 1] 目标推力 (N)
                           仅 kinematic 模式使用
            
        Returns:
            effort: [batch, n_drones] 总推力指标
        """
        if self.sim_mode == "kinematic":
            return self._apply_action_kinematic(target_rate, target_thrust)
        else:
            return self._apply_action_physics(actions)
    
    def _apply_action_physics(self, actions: torch.Tensor) -> torch.Tensor:
        """原有的物理引擎模式"""
        rotor_cmds = actions.expand(*self.shape, self.num_rotors)
        last_throttle = self.throttle.clone()
        thrusts, moments = vmap(vmap(self.rotors, randomness="different"), randomness="same")(
            rotor_cmds, self.rotor_params
        )

        rotor_pos, rotor_rot = self.rotors_view.get_world_poses()
        torque_axis = quat_axis(rotor_rot.flatten(end_dim=-2), axis=2).unflatten(0, (*self.shape, self.num_rotors))

        self.thrusts[..., 2] = thrusts
        self.torques[:] = (moments.unsqueeze(-1) * torque_axis).sum(-2)
        # TODO@btx0424: general rotating rotor
        if self.is_articulation and self.rotor_joint_indices is not None:
            rot_vel = (self.throttle * self.directions * self.MAX_ROT_VEL)
            self._view.set_joint_velocities(
                rot_vel.reshape(-1, self.num_rotors),
                joint_indices=self.rotor_joint_indices
            )
        self.forces.zero_()
        # TODO: global downwash
        # if self.n > 1:
        #     self.forces[:] += vmap(self.downwash)(
        #         self.pos,
        #         self.pos,
        #         quat_rotate(self.rot, self.thrusts.sum(-2)),
        #         kz=0.3
        #     ).sum(-2)
        velocity_norm = torch.norm(self.vel[..., :3], dim=-1, keepdim=True)
        self.forces[:] -= self.drag_coef * velocity_norm * self.vel[..., :3]

        # print("thrusts in apply_action:", thrusts)
        self.rotors_view.apply_forces_and_torques_at_pos(
            self.thrusts.reshape(-1, 3),
            is_global=False
        )
        self.base_link.apply_forces_and_torques_at_pos(
            self.forces.reshape(-1, 3),
            self.torques.reshape(-1, 3),
            is_global=True
        )
        self.throttle_difference[:] = torch.norm(self.throttle - last_throttle, dim=-1)
        return self.throttle.sum(-1)
    
    def _apply_action_kinematic(self, target_rate: torch.Tensor, target_thrust: torch.Tensor) -> torch.Tensor:
        """
        使用内部动力学模型更新状态，绕过 Isaac Sim 物理引擎
        
        直接接收物理控制量，无需经过电机模型和 RateController
        这确保了与 MPC 预测模型的完全一致性
        
        Args:
            target_rate: [batch, n_drones, 3] 目标角速度 (rad/s)
            target_thrust: [batch, n_drones, 1] 目标推力 (N)
            
        Returns:
            effort: [batch, n_drones] 总推力（返回 target_thrust 以保持接口一致）
        """
        # 1. 获取当前状态
        pos, rot = self.get_world_poses(clone=False)  # [batch, n_drones, 3/4]
        vel = self.get_velocities(clone=False)  # [batch, n_drones, 6]
        
        batch_size = pos.shape[0]
        n_drones = pos.shape[1]
        
        # 2. 构造动力学模型的状态向量 x [total_drones, 10]
        pos_flat = pos.reshape(-1, 3)
        rot_flat = rot.reshape(-1, 4)  # Isaac Sim: (w, x, y, z)
        lin_vel_flat = vel[..., :3].reshape(-1, 3)
        
        current_state = torch.cat([pos_flat, rot_flat, lin_vel_flat], dim=1)
        
        # 3. 准备控制输入 u [total_drones, 4]
        # u = [thrust_force, wx, wy, wz]
        thrust_flat = target_thrust.reshape(-1, 1)  # [total_drones, 1]
        omega_flat = target_rate.reshape(-1, 3)     # [total_drones, 3]
        
        u_flat = torch.cat([thrust_flat, omega_flat], dim=1)  # [total_drones, 4]
        
        # 4. 动力学积分
        next_state = self.dynamics_model(current_state, u_flat)
        
        # 5. 解析并设置新状态
        next_pos = next_state[:, :3].reshape(batch_size, n_drones, 3)
        next_rot = next_state[:, 3:7].reshape(batch_size, n_drones, 4)
        next_lin_vel = next_state[:, 7:10].reshape(batch_size, n_drones, 3)
        
        # 6. 强制设置 Isaac Sim 中的位置和姿态 (运动学更新)
        self.set_world_poses(next_pos, next_rot)
        
        # 7. 设置速度 (线速度 + 角速度)
        # 确保 target_rate 有正确的形状 [batch_size, n_drones, 3]
        target_rate_reshaped = target_rate.reshape(batch_size, n_drones, 3)
        full_vel = torch.cat([
            next_lin_vel,
            target_rate_reshaped  # 使用 reshape 后的角速度
        ], dim=-1)
        self.set_velocities(full_vel)
        
        # 8. 更新内部状态记录（保持接口兼容性）
        self.forces[:] = 0.0  # 运动学模式不施加物理力
        self.torques[:] = 0.0
        
        # 返回总推力 (保持接口一致)
        return target_thrust.squeeze(-1)  # [batch, n_drones]

    def get_state(self, check_nan: bool=False, env_frame: bool=True):
        self.pos[:], self.rot[:] = self.get_world_poses(True)
        if env_frame and hasattr(self, "_envs_positions"):
            self.pos.sub_(self._envs_positions)

        vel_w = self.get_velocities(True)
        vel_b = torch.cat([
            quat_rotate_inverse(self.rot, vel_w[..., :3]),
            quat_rotate_inverse(self.rot, vel_w[..., 3:])
        ], dim=-1)
        self.vel_w[:] = vel_w
        self.vel_b[:] = vel_b

        # acc = self.acc.lerp((vel - self.vel) / self.dt, self.alpha)
        # self.acc[:] = acc
        self.heading[:] = quat_axis(self.rot, axis=0)
        self.up[:] = quat_axis(self.rot, axis=2)
        state = [self.pos, self.rot, self.vel, self.heading, self.up, self.throttle * 2 - 1]

        state = torch.cat(state, dim=-1)
        if check_nan:
            assert not torch.isnan(state).any()
        return state

    def _reset_idx(self, env_ids: torch.Tensor, train: bool=True):
        if env_ids is None:
            env_ids = torch.arange(self.shape[0], device=self.device)
        self.thrusts[env_ids] = 0.0
        self.torques[env_ids] = 0.0
        self.vel[env_ids] = 0.
        self.acc[env_ids] = 0.
        # self.jerk[env_ids] = 0.
        if train and "train" in self.randomization:
            self._randomize(env_ids, self.randomization["train"])
        elif "eval" in self.randomization:
            self._randomize(env_ids, self.randomization["eval"])
        init_throttle = self.gravity[env_ids] / self.KF[env_ids].sum(-1, keepdim=True)
        self.throttle[env_ids] = self.rotors.f_inv(init_throttle)
        self.throttle_difference[env_ids] = 0.0
        return env_ids

    def _randomize(self, env_ids: torch.Tensor, distributions: Dict[str, D.Distribution]):
        shape = env_ids.shape
        if "mass" in distributions:
            masses = distributions["mass"].sample(shape)
            self.base_link.set_masses(masses, env_indices=env_ids)
            self.masses[env_ids] = masses
            self.gravity[env_ids] = masses * 9.81
            self.intrinsics["mass"][env_ids] = (masses / self.MASS_0)
        if "inertia" in distributions:
            inertias = distributions["inertia"].sample(shape)
            self.inertias[env_ids] = inertias
            self.base_link.set_inertias(
                torch.diag_embed(inertias).flatten(-2), env_indices=env_ids
            )
            self.intrinsics["inertia"][env_ids] = inertias / self.INERTIA_0
        if "com" in distributions:
            coms = distributions["com"].sample((*shape, 3))
            self.base_link.set_coms(coms, env_indices=env_ids)
            self.intrinsics["com"][env_ids] = coms.reshape(*shape, 1, 3)
        if "thrust2weight" in distributions:
            thrust2weight = distributions["thrust2weight"].sample(shape)
            KF = thrust2weight * self.masses[env_ids] * 9.81
            self.KF[env_ids] = KF
            self.intrinsics["KF"][env_ids] = KF / self.KF_0
        if "force2moment" in distributions:
            force2moment = distributions["force2moment"].sample(shape)
            KM = self.KF[env_ids] / force2moment
            self.KM[env_ids] = KM
            self.intrinsics["KM"][env_ids] = KM / self.KM_0
        if "drag_coef" in distributions:
            drag_coef = distributions["drag_coef"].sample(shape).reshape(-1, 1, 1)
            self.drag_coef[env_ids] = drag_coef
            self.intrinsics["drag_coef"][env_ids] = drag_coef
        if "tau_up" in distributions:
            tau_up = distributions["tau_up"].sample(shape+self.rotors_view.shape[1:])
            self.tau_up[env_ids] = tau_up
            self.intrinsics["tau_up"][env_ids] = tau_up
        if "tau_down" in distributions:
            tau_down = distributions["tau_down"].sample(shape+self.rotors_view.shape[1:])
            self.tau_down[env_ids] = tau_down
            self.intrinsics["tau_down"][env_ids] = tau_down

    def get_thrust_to_weight_ratio(self):
        return self.KF.sum(-1, keepdim=True) / (self.masses * 9.81)

    def get_linear_smoothness(self):
        return - (
            torch.norm(self.acc[..., :3], dim=-1)
            + torch.norm(self.jerk[..., :3], dim=-1)
        )

    def get_angular_smoothness(self):
        return - (
            torch.sum(self.acc[..., 3:].abs(), dim=-1)
            + torch.sum(self.jerk[..., 3:].abs(), dim=-1)
        )

    def __str__(self):
        default_params = "\n".join([
            "Default parameters:",
            f"Mass: {self.MASS_0.tolist()}",
            f"Inertia: {self.INERTIA_0.tolist()}",
            f"Thrust2Weight: {self.THRUST2WEIGHT_0.tolist()}",
            f"Force2Moment: {self.FORCE2MOMENT_0.tolist()}",
        ])
        return default_params

    @staticmethod
    def downwash(
        p0: torch.Tensor,
        p1: torch.Tensor,
        p1_t: torch.Tensor,
        kr: float=2,
        kz: float=1,
    ):
        """
        A highly simplified downwash effect model.

        References:
        https://arxiv.org/pdf/2207.09645.pdf
        https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8798116

        """
        z, r = separation(p0, p1, normalize(p1_t))
        z = torch.clip(z, 0)
        v = torch.exp(-0.5 * torch.square(kr * r / z)) / (1 + kz * z)**2
        f = off_diag(v * - p1_t)
        return f

    @staticmethod
    def _coerce_robot_cfg(cfg, cfg_cls):
        """Convert user-provided config (dict / OmegaConf / dataclass) into cfg_cls."""
        if cfg is None or isinstance(cfg, cfg_cls):
            return cfg

        # lazily import OmegaConf to avoid hard dependency for users without hydra
        try:
            from omegaconf import DictConfig, OmegaConf  # type: ignore
        except ImportError:  # pragma: no cover - optional dependency
            DictConfig = ()  # type: ignore
            OmegaConf = None  # type: ignore

        if "DictConfig" in locals() and isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)

        if isinstance(cfg, dict):
            cfg_obj = copy.deepcopy(cfg_cls())
            MultirotorBase._deep_update_dataclass(cfg_obj, cfg)
            return cfg_obj

        return cfg

    @staticmethod
    def _deep_update_dataclass(target, updates):
        for key, value in updates.items():
            if hasattr(target, key):
                attr = getattr(target, key)
                if is_dataclass(attr) and isinstance(value, dict):
                    MultirotorBase._deep_update_dataclass(attr, value)
                    continue
            setattr(target, key, value)

    @staticmethod
    def make(drone_model: str, controller: str=None, device: str="cpu", cfg: dict=None):
        """
        创建无人机实例和控制器
        
        Args:
            drone_model: 无人机模型名称 (e.g., "Crazyflie", "Hummingbird")
            controller: 控制器名称 (e.g., "RateController", "PIDRateController", None)
            device: 设备 ("cpu" or "cuda")
            cfg: 配置字典，可包含 sim_mode 等参数
            
        Returns:
            (drone, controller): 无人机实例和控制器实例
        """
        drone_cls = MultirotorBase.REGISTRY[drone_model]
        robot_cfg = MultirotorBase._coerce_robot_cfg(cfg, drone_cls.cfg_cls)
        
        # Kinematic 模式下必须禁用 Isaac Sim 的重力（动力学模型已经包含重力效应）
        if cfg and isinstance(cfg, dict) and cfg.get("sim_mode") == "kinematic":
            if robot_cfg is None:
                robot_cfg = drone_cls.cfg_cls()
            robot_cfg.rigid_props.disable_gravity = True
            logging.info("[MultirotorBase.make] Kinematic mode detected: disabling Isaac Sim gravity")
        
        # Generate unique name from cfg or use model name with timestamp
        import time
        drone_name = None
        
        # 检查 cfg 字典中是否明确指定了 name
        if cfg and isinstance(cfg, dict) and "name" in cfg and cfg["name"]:
            drone_name = cfg["name"]
        
        # 如果没有明确的名称，生成唯一名称（使用时间戳避免冲突）
        if not drone_name:
            unique_suffix = f"_{int(time.time() * 1000000) % 1000000}"
            drone_name = f"{drone_model}{unique_suffix}"
        
        # 传递配置到无人机构造函数
        drone = drone_cls(name=drone_name, cfg=robot_cfg) if robot_cfg is not None else drone_cls(name=drone_name)
        from omni_drones.controllers import ControllerBase
        if controller is not None and controller != "RotorController":
            controller_cls = ControllerBase.REGISTRY[controller]
            if controller == "PIDRateController":
                controller = controller_cls(drone.dt, drone.gravity[1], drone.params).to(device)
            else:
                controller = controller_cls(drone.gravity[1], drone.params).to(device)
        else:
            # When no controller is specified or it's "RotorController", return None for controller
            controller = None
        return drone, controller


def separation(p0, p1, p1_d):
    rel_pos = rel_pos =  p1.unsqueeze(0) - p0.unsqueeze(1)
    z_distance = (rel_pos * p1_d).sum(-1, keepdim=True)
    z_displacement = z_distance * p1_d

    r_displacement = rel_pos - z_displacement
    r_distance = torch.norm(r_displacement, dim=-1, keepdim=True)
    return z_distance, r_distance

