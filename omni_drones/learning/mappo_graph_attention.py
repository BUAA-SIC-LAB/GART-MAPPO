# MIT License
#
# Copyright (c) 2025
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
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

"""Graph-Attention MAPPO variant with ego-graph, task attention, and GRU memory."""

from __future__ import annotations

import datetime
import time
from typing import Dict, Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, cast
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from .modules.distributions import IndependentNormal
from .modules.graph_recurrent_attention import (
    GraphRecurrentAttentionConfig,
    GraphRecurrentAttentionEncoder,
)
from .ppo.common import GAE, make_mlp1, make_mlp
from .utils import valuenorm
from .utils.valuenorm import ValueNorm1
from torch._functorch.apis import vmap
from einops.layers.torch import Rearrange

class Actor(nn.Module):
    def __init__(self, action_dim: int, predict_std: bool=False) -> None:
        super().__init__()
        self.predict_std = predict_std
        if predict_std:
            self.actor_mean = nn.Tanh(nn.LazyLinear(action_dim * 2))
        else:
            self.actor_mean = nn.LazyLinear(action_dim)
            self.actor_std = nn.Parameter(torch.zeros(action_dim))
        self.scale_mapping = torch.exp

    def forward(self, features: torch.Tensor):
        if self.predict_std:
            loc, scale = self.actor_mean(features).chunk(2, dim=-1)
        else:
            loc = self.actor_mean(features)
            scale = self.actor_std.expand_as(loc)
        scale = self.scale_mapping(scale)
        return loc, scale


class EnsembleModule(_EnsembleModule):

    def __init__(self, module: TensorDictModuleBase, num_copies: int):
        super(_EnsembleModule, self).__init__()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.num_copies = num_copies

        params_td = make_functional(module).expand(num_copies).to_tensordict()
        self.module = module
        self.vmapped_forward = vmap(self.module, (1, 0), 1)
        # self.reset_parameters_recursive(params_td)
        self.params_td = TensorDictParams(params_td)

    def forward(self, tensordict: TensorDict):
        tensordict = tensordict.select(*self.in_keys)
        tensordict.batch_size = [tensordict.shape[0], self.num_copies]
        return self.vmapped_forward(tensordict, self.params_td)


def init_(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, 1.414)
        nn.init.constant_(module.bias, 0)

class EncoderWrapper(nn.Module):
    """Wrapper to maintain encoder output in [B, N, D] format."""
    
    def __init__(self, encoder: GraphRecurrentAttentionEncoder, num_agents: int):
        super().__init__()
        self.encoder = encoder
        self.num_agents = num_agents
        self.current_stage = 0  # Default stage
        
    def forward(self, observation: TensorDict, is_init: torch.Tensor, memory: torch.Tensor):
        # curriculum_stage is stored as an instance variable, not passed through tensordict
        # This avoids batch dimension mismatch issues
        features, updated_hidden = self.encoder(observation, is_init, memory, curriculum_stage=self.current_stage)
        # Keep both tensors in their original [B, N, D] format
        # This ensures batch_size matches the input TensorDict batch_size
        return features, updated_hidden
    
    def set_curriculum_stage(self, stage: int):
        """Update the curriculum stage."""
        self.current_stage = stage


class ActorWrapper(nn.Module):
    """Wrapper to flatten latent features, apply actor, and reshape output."""
    
    def __init__(self, actor_trunk: nn.Module, num_agents: int, action_dim: int):
        super().__init__()
        self.actor_trunk = actor_trunk
        self.num_agents = num_agents
        self.action_dim = action_dim
        
    def forward(self, latent_feature: torch.Tensor):
        # latent_feature: Can be [B, N, D] or [B, T, N, D]
        
        # Store original shape info
        batch_dims = latent_feature.shape[:-2] # [B] or [B, T]
        num_agents = latent_feature.shape[-2]
        
        # Flatten to [B*T*N, D] or [B*N, D] for actor processing
        latent_flat = latent_feature.reshape(-1, latent_feature.shape[-1])
        
        # actor_trunk output: tuple (loc, scale) each [B*T*N, action_dim]
        loc, scale = self.actor_trunk(latent_flat)
        
        # Reshape back to original batch shape
        loc = loc.reshape(*batch_dims, num_agents, self.action_dim)
        scale = scale.reshape(*batch_dims, num_agents, self.action_dim)
        return loc, scale

##### CHANGED #####
# This is the correct batching function for TBPTT
def make_batch(tensordict: TensorDict, num_minibatches: int, seq_len: int):
    """
    Creates an iterator of minibatches of sequences from a full rollout.
    """
    if seq_len > 1:
        # We have B=num_envs, T=rollout_length
        N, T = tensordict.shape
        T = (T // seq_len) * seq_len
        if T == 0:
            raise RuntimeError(f"Rollout length {tensordict.shape[1]} is shorter than seq_len {seq_len}")
        
        # Reshape from [B, T_rollout, ...] to [B * (T_rollout / seq_len), seq_len, ...]
        tensordict = tensordict[:, :T].reshape(-1, seq_len)
        
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]
    else:
        # Fallback for i.i.d. sampling (seq_len=1)
        tensordict = tensordict.reshape(-1)
        perm = torch.randperm(
            (tensordict.shape[0] // num_minibatches) * num_minibatches,
            device=tensordict.device,
        ).reshape(num_minibatches, -1)
        for indices in perm:
            yield tensordict[indices]


class GraphAttentionMAPPO:
    def __init__(
        self,
        cfg,
        observation_spec: CompositeSpec,
        action_spec: TensorSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # Curriculum learning tracking
        self.current_curriculum_stage = 0
        
        # Check if actor should be shared across agents
        self.share_actor = getattr(cfg, "share_actor", True)
        
        ##### CHANGED #####
        # Add seq_len parameter, defaulting to 8 if not specified
        self.seq_len = getattr(cfg, "seq_len", 8)
        
        self.entropy_coef = getattr(cfg, "entropy_coef", 0.001)
        self.clip_param = getattr(cfg, "clip_param", 0.1)
        self.policy_loss_weight = getattr(cfg, "policy_loss_weight", 1.0)
        self.value_loss_weight = getattr(cfg, "value_loss_weight", 1.0)
        huber_delta = getattr(cfg.critic, "huber_delta", 10.0)
        self.critic_loss_fn = nn.HuberLoss(delta=huber_delta, reduction="none") # ##### CHANGED #####: Use reduction="none" for masking
        gamma = getattr(cfg, "gamma", 0.995)
        gae_lambda = getattr(cfg, "gae_lambda", 0.95)
        self.gae = GAE(gamma, gae_lambda)

        if action_spec.ndim <= 2:
            raise ValueError("GraphAttentionMAPPO expects multi-agent action spec.")

        self.num_agents, self.action_dim = action_spec.shape[-2:]

        obs_agent_spec: CompositeSpec = observation_spec[("agents", "observation")]
        obs_central_spec: CompositeSpec = observation_spec[("agents", "observation_central")]

        model_cfg = getattr(cfg, "model")
        encoder_cfg = GraphRecurrentAttentionConfig(
            embed_dim=model_cfg.embed_dim,
            gnn_heads=model_cfg.gnn_heads,
            gnn_layers=model_cfg.gnn_layers,
            gnn_dropout=model_cfg.gnn_dropout,
            task_heads=model_cfg.task_heads,
            fusion_hidden_dim=model_cfg.fusion_hidden_dim,
            gru_hidden_dim=model_cfg.gru_hidden_dim,
            post_mlp_units=tuple(model_cfg.post_mlp_units),
            use_graph=getattr(model_cfg, "use_graph", True),
            use_task_attn=getattr(model_cfg, "use_task_attn", True),
            use_gru=getattr(model_cfg, "use_gru", True),
        )

        ##### CHANGED #####
        # Create independent encoders for actor and critic to avoid gradient conflicts
        self.actor_encoder = GraphRecurrentAttentionEncoder(obs_agent_spec, encoder_cfg).to(self.device)
        
        actor_encoder_wrapper = EncoderWrapper(self.actor_encoder, self.num_agents)
        # Store reference to wrapper so we can update its curriculum stage
        self.actor_encoder_wrapper = actor_encoder_wrapper

        actor_hidden_units = list(model_cfg.actor_head_units)
        if not actor_hidden_units:
            raise ValueError("actor_head_units must contain at least one element")
        
        # Use make_mlp1 which automatically adds Tanh() to the last layer
        actor_mlp = make_mlp1(actor_hidden_units, activation=nn.Mish)
        actor_trunk = nn.Sequential(actor_mlp, Actor(self.action_dim, predict_std=False)).to(self.device)
        
        ##### CHANGED #####
        # ActorWrapper must now handle [B, T, N, D] inputs
        actor_wrapper = ActorWrapper(actor_trunk, self.num_agents, self.action_dim)

        # Actor encoder module with separate memory key
        # curriculum_stage is optional and will be added during forward pass
        self.actor_encoder_module = TensorDictModule(
            actor_encoder_wrapper,
            in_keys=[("agents", "observation"), "is_init", "actor_memory"],
            out_keys=["actor_latent_feature", ("next", "actor_memory")],
        )

        # Create actor body
        actor_body = TensorDictModule(actor_wrapper, ["actor_latent_feature"], ["loc", "scale"])
        
        # Initialize lazy modules in actor_body before applying EnsembleModule
        # This is required because EnsembleModule calls make_functional which needs initialized parameters
        fake_input = observation_spec.zero().to(self.device)
        self._maybe_init_actor_memory(fake_input)
        fake_input.set("is_init", torch.zeros(fake_input.batch_size + (1,), dtype=torch.bool, device=self.device))
        # Don't add curriculum_stage to fake_input during initialization - it will be added during forward pass
        # The encoder wrapper will handle None curriculum_stage gracefully
        actor_encoded = self.actor_encoder_module(fake_input)
        fake_input.update(actor_encoded)
        
        # Forward pass to initialize lazy layers
        _ = actor_body(fake_input)
        
        # Now apply ensemble if not sharing
        if not self.share_actor:
            print(f"Creating separate actors for each of {self.num_agents} agents using EnsembleModule")
            actor_body = EnsembleModule(actor_body, self.num_agents)
        else:
            print(f"Sharing single actor across all {self.num_agents} agents")
        
        self.actor_body = actor_body
        self.actor = ProbabilisticActor(
            module=cast(Any, self.actor_body),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        # ✅ Critic: 前馈 MLP 直接处理中心化观测（参考 MAPPO）
        critic_hidden_units = list(model_cfg.critic_head_units)
        if not critic_hidden_units:
            raise ValueError("critic_head_units must contain at least one element")
        
        # 构建 Critic 网络：需要先展平 observation_central 的嵌套结构
        # observation_central 包含 {"drones": ..., "gates": ..., "formation": ...}
        class CriticNet(nn.Module):
            def __init__(self, hidden_units, num_agents, use_popart=False):
                super().__init__()
                self.mlp = make_mlp(hidden_units, activation=nn.Mish)
                self.use_popart = use_popart
                self.num_agents = num_agents
                
                if use_popart:
                    # PopArt will replace the output layer after initialization
                    # Store lazy linear temporarily for initialization
                    self.lazy_output: nn.Module = nn.LazyLinear(num_agents)
                else:
                    self.lazy_output: nn.Module = nn.LazyLinear(num_agents)
                
                # v_out will be set to either the lazy_output or PopArt layer after initialization
                self.v_out: nn.Module = self.lazy_output
                
            def forward(self, obs_central):
                # obs_central 是一个 TensorDict，需要先展平
                # 提取所有观测并展平
                drones = obs_central["drones"]  # [B, N, drone_dim]
                gates = obs_central["gates"]    # [B, K, gate_dim]
                formation = obs_central["formation"]  # [B, N, 3]
                
                # 展平所有维度
                drones_flat = einops.rearrange(drones, "... n d -> ... (n d)")
                gates_flat = einops.rearrange(gates, "... k d -> ... (k d)")
                formation_flat = einops.rearrange(formation, "... n d -> ... (n d)")
                
                # 拼接所有特征
                features = torch.cat([drones_flat, gates_flat, formation_flat], dim=-1)
                
                # MLP 处理
                x = self.mlp(features)
                
                # Output layer (either PopArt or standard Linear)
                x = self.v_out(x)
                
                # 添加值维度 [B, N] -> [B, N, 1]
                return x.unsqueeze(-1)
        
        # Check if PopArt is enabled in config
        use_popart = getattr(cfg, "use_popart", False)
        
        self.critic_net = CriticNet(critic_hidden_units, self.num_agents, use_popart=use_popart)
        self.critic_module = TensorDictModule(
            self.critic_net,
            [("agents", "observation_central")], 
            ["state_value"]
        ).to(self.device)

        # Independent optimizers for actor and critic
        actor_lr = getattr(cfg.actor, "lr", 3e-4)
        critic_lr = getattr(cfg.critic, "lr", 3e-4)
        
        self.actor_opt = torch.optim.Adam(
            list(self.actor_encoder.parameters()) + list(self.actor.parameters()),
            lr=actor_lr
        )
        self.critic_opt = torch.optim.Adam(
            self.critic_module.parameters(),
            lr=critic_lr
        )

        # Initialize lazy modules in critic
        self.critic_module(fake_input)
        
        # Now replace lazy linear with PopArt if enabled
        if use_popart:
            # Get the input dimension from the initialized lazy linear
            input_dim = self.critic_net.lazy_output.in_features
            popart_beta = getattr(cfg, "popart_beta", 0.9995)
            
            # Create PopArt layer
            self.critic_net.v_out = valuenorm.PopArt(
                input_shape=input_dim,
                output_shape=self.num_agents,
                beta=popart_beta
            ).to(self.device)
            
            # Initialize PopArt weights from the lazy linear weights
            with torch.no_grad():
                self.critic_net.v_out.weight.copy_(self.critic_net.lazy_output.weight)
                self.critic_net.v_out.bias.copy_(self.critic_net.lazy_output.bias)
            
            # IMPORTANT: PopArt is the output layer itself, so we use a separate
            # ValueNorm1 wrapper for denormalization with input_shape matching
            # the critic output shape [B, N, 1]
            self.value_normalizer = ValueNorm1(input_shape=1, beta=popart_beta).to(self.device)
            self.use_popart = True
            print(f"Using PopArt output layer with beta={popart_beta}")
        elif hasattr(cfg, "value_norm") and cfg.value_norm is not None:
            # Alternative: use value_norm config with dynamic class loading
            cls = getattr(valuenorm, cfg.value_norm["class"])
            kwargs = cfg.value_norm.get("kwargs", {})
            self.value_normalizer: valuenorm.Normalizer = cls(input_shape=1, **kwargs).to(self.device)
            self.use_popart = False
            print(f"Using {cfg.value_norm['class']} normalization")
        else:
            # Default: use ValueNorm1
            self.value_normalizer = ValueNorm1(input_shape=1).to(self.device)
            self.use_popart = False
            print("Using ValueNorm1 normalization")
        
        # Re-initialize critic optimizer to include PopArt parameters if needed
        if use_popart:
            self.critic_opt = torch.optim.Adam(
                self.critic_module.parameters(),
                lr=critic_lr
            )
        
        # Test actor (already initialized)
        self.actor(fake_input)

    def _maybe_init_actor_memory(self, tensordict: TensorDictBase):
        """Initialize actor memory if not present."""
        if tensordict.get("actor_memory", None) is not None:
            return tensordict
        obs_self = tensordict.get(("agents", "observation", "obs_self"))
        batch_shape = obs_self.shape[:-1]
        
        agent_dim = -3
        batch_shape = obs_self.shape[:agent_dim]
        
        memory_shape = batch_shape + (self.num_agents,) + (self.actor_encoder.cfg.gru_hidden_dim,)
        
        tensordict.set(
            "actor_memory",
            torch.zeros(*memory_shape, device=self.device, dtype=obs_self.dtype),
        )
        return tensordict

    def __call__(self, tensordict: TensorDict):
        """
        Forward pass for rollout (single timestep).
        
        Args:
            tensordict: Single timestep data [B, ...]
        
        Returns:
            tensordict with actions and values
        """
        # Initialize actor memory if needed
        self._maybe_init_actor_memory(tensordict)
        if tensordict.get("is_init", None) is None:
            init_shape = tensordict.batch_size + (1,)
            tensordict.set(
                "is_init",
                torch.zeros(*init_shape, dtype=torch.bool, device=self.device),
            )
        
        # curriculum_stage is managed by actor_encoder_wrapper internally
        # No need to add it to tensordict
        
        # Start timing
        t0 = time.perf_counter()

        # Encode with actor encoder
        encode_out = self.actor_encoder_module(tensordict)
        tensordict.update(encode_out)
        
        next_memory = encode_out.get(("next", "actor_memory"), None)
        if next_memory is not None:
            tensordict.set("actor_memory", next_memory)
        
        # Actor forward
        tensordict.update(self.actor(tensordict))

        # End timing
        t1 = time.perf_counter()
        inference_time = t1 - t0
        
        # Add inference time to tensordict
        tensordict.set("inference_time", torch.full(tensordict.shape, inference_time, device=self.device, dtype=torch.float32))
        
        # Critic forward (直接使用中心化观测，无需编码器)
        tensordict.update(self.critic_module(tensordict))
        
        # Clean up intermediate features
        tensordict.exclude("actor_latent_feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict) -> Dict[str, float]:
        """
        Compute GAE and prepare data for training.
        Critic uses feedforward network, so no vmap encoding needed.
        """
        next_tensordict = tensordict["next"]
        
        with torch.no_grad():
            # ✅ Critic 直接前向传播，无需编码器
            next_values = self.critic_module(next_tensordict)["state_value"]
        
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]
        dones = einops.repeat(dones, "b t 1 -> b t a 1", a=self.num_agents)
        values = tensordict["state_value"]
        values = self.value_normalizer.denormalize(values)
        next_values = self.value_normalizer.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        
        # Update both PopArt layer (for weight rescaling) and ValueNorm1 (for normalization)
        if self.use_popart:
            # Update PopArt layer's statistics (this rescales the output weights)
            # PopArt expects input without the extra dimension, so we squeeze
            ret_for_popart = ret.squeeze(-1)  # [B, T, N, 1] -> [B, T, N]
            self.critic_net.v_out.update(ret_for_popart)
        
        # Update ValueNorm1 for normalization
        self.value_normalizer.update(ret)
        ret = self.value_normalizer.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _ in range(self.cfg.ppo_epochs):
            ##### CHANGED #####
            # Call the new make_batch with seq_len
            batch_iter = make_batch(tensordict, self.cfg.num_minibatches, self.seq_len)
            
            for minibatch in batch_iter:
                infos.append(self._update(minibatch))

        if infos:
            # Stack all info dicts and compute mean
            stacked = {}
            for key in infos[0].keys():
                values = torch.stack([info[key] for info in infos])
                stacked[key] = values.mean()
            return {k: v.item() for k, v in stacked.items()}
        return {}

    def _update(self, tensordict: TensorDict) -> TensorDict:
        """Perform one PPO update over a sequence minibatch."""

        B_mini, T_seq = tensordict.shape[:2]

        # ✅ 使用序列第一个时间步的 actor_memory 作为初始 hidden state
        # 这确保了 TBPTT 重现 rollout 时的行为
        first_step = tensordict[:, 0]
        actor_hidden = first_step.get("actor_memory")  # [B_mini, N, hidden_dim]
        
        if actor_hidden is None:
            # Fallback: 只有在没有保存 memory 时才使用零初始化
            obs_self_seq = tensordict.get(("agents", "observation", "obs_self"))
            actor_hidden_dim = self.actor_encoder.cfg.gru_hidden_dim
            actor_hidden = obs_self_seq.new_zeros((B_mini, self.num_agents, actor_hidden_dim))

        all_log_probs: list[torch.Tensor] = []
        all_entropies: list[torch.Tensor] = []

        for t in range(T_seq):
            step_td = tensordict[:, t]
            obs_t: TensorDict = step_td.get(("agents", "observation"))
            is_init_t = step_td.get("is_init")

            actor_features_t, actor_hidden = self.actor_encoder(
                obs_t,
                is_init=is_init_t,
                hidden_state=actor_hidden,
            )

            actor_td = TensorDict(
                {"actor_latent_feature": actor_features_t},
                batch_size=actor_features_t.shape[:-1],
                device=actor_features_t.device,
            )
            dist_t = self.actor.get_dist(actor_td)
            actions_t = step_td[("agents", "action")]
            log_prob_t = dist_t.log_prob(actions_t)
            entropy_t = dist_t.entropy()

            all_log_probs.append(log_prob_t)
            all_entropies.append(entropy_t)

        log_probs = torch.stack(all_log_probs, dim=1)  # [B_mini, T_seq, N]
        entropy = torch.stack(all_entropies, dim=1)    # [B_mini, T_seq, N]

        # ✅ Critic 前向传播（无需 TBPTT）
        # 直接使用中心化观测
        values = self.critic_module(tensordict)["state_value"]

        old_log_prob = tensordict["sample_log_prob"].reshape_as(log_probs)
        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - old_log_prob).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim * self.policy_loss_weight
        entropy_loss = -self.entropy_coef * entropy.mean()

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped).mean() * self.value_loss_weight

        loss = policy_loss + entropy_loss + value_loss

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()

        max_grad_norm = getattr(self.cfg, "max_grad_norm", None)
        actor_params = list(self.actor_encoder.parameters()) + list(self.actor.parameters())
        critic_params = list(self.critic_module.parameters())

        if max_grad_norm is not None and max_grad_norm > 0:
            actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(actor_params, max_grad_norm)
            critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(critic_params, max_grad_norm)
        else:
            actor_grad_norm = torch.linalg.norm(
                torch.stack([p.grad.detach().norm() for p in actor_params if p.grad is not None])
            ) if any(p.grad is not None for p in actor_params) else torch.tensor(0.0, device=loss.device)
            critic_grad_norm = torch.linalg.norm(
                torch.stack([p.grad.detach().norm() for p in critic_params if p.grad is not None])
            ) if any(p.grad is not None for p in critic_params) else torch.tensor(0.0, device=loss.device)

        self.actor_opt.step()
        self.critic_opt.step()

        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()

        metrics = {
            "loss": loss.detach(),
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach(),
            "entropy_loss": entropy_loss.detach(),
            "entropy": entropy.mean().detach(),
            "explained_var": explained_var.detach(),
            "actor_grad_norm": actor_grad_norm.detach() if isinstance(actor_grad_norm, torch.Tensor) else torch.tensor(actor_grad_norm),
            "critic_grad_norm": critic_grad_norm.detach() if isinstance(critic_grad_norm, torch.Tensor) else torch.tensor(critic_grad_norm),
        }

        return TensorDict(metrics, [])

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic_module.state_dict(),
            "actor_encoder": self.actor_encoder.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "value_normalizer": self.value_normalizer.state_dict(),
            "num_agents": self.num_agents,
            "action_dim": self.action_dim,
            "entropy_coef": self.entropy_coef,
            "clip_param": self.clip_param,
            "policy_loss_weight": self.policy_loss_weight,
            "value_loss_weight": self.value_loss_weight,
            "share_actor": self.share_actor,
        }

    def load_state_dict(self, state_dict):
        if state_dict.get("num_agents", self.num_agents) != self.num_agents:
            raise ValueError("Mismatch in num_agents for loaded checkpoint")
        if state_dict.get("action_dim", self.action_dim) != self.action_dim:
            raise ValueError("Mismatch in action_dim for loaded checkpoint")
        
        # Check share_actor compatibility
        loaded_share_actor = state_dict.get("share_actor", True)
        if loaded_share_actor != self.share_actor:
            print(f"Warning: share_actor mismatch - checkpoint: {loaded_share_actor}, current: {self.share_actor}")
            print("This may cause issues if model architectures differ")
        
        self.actor.load_state_dict(state_dict["actor"])
        self.critic_module.load_state_dict(state_dict["critic"])
        
        # Load actor encoder
        if "actor_encoder" in state_dict:
            self.actor_encoder.load_state_dict(state_dict["actor_encoder"])
        elif "encoder" in state_dict:
            # Backward compatibility: old checkpoints may have shared encoder
            self.actor_encoder.load_state_dict(state_dict["encoder"])
        
        self.actor_opt.load_state_dict(state_dict["actor_opt"])
        self.critic_opt.load_state_dict(state_dict["critic_opt"])
        
        # Load value normalizer with backward compatibility
        if "value_normalizer" in state_dict:
            self.value_normalizer.load_state_dict(state_dict["value_normalizer"])
        elif "value_norm" in state_dict:
            # Backward compatibility
            self.value_normalizer.load_state_dict(state_dict["value_norm"])
        
        self.entropy_coef = state_dict.get("entropy_coef", self.entropy_coef)
        self.clip_param = state_dict.get("clip_param", self.clip_param)
        self.policy_loss_weight = state_dict.get("policy_loss_weight", self.policy_loss_weight)
        self.value_loss_weight = state_dict.get("value_loss_weight", self.value_loss_weight)

    def save_checkpoint(self, filepath: str):
        checkpoint = {"model_state_dict": self.state_dict(), "timestamp": time.time()}
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint loaded from {filepath}")
        if "timestamp" in checkpoint:
            ts = datetime.datetime.fromtimestamp(checkpoint["timestamp"])
            print(f"Checkpoint created at: {ts}")
    
    def update_curriculum_stage(self, new_stage: int):
        """Update curriculum learning stage and reset optimizers if stage changed.
        
        Args:
            new_stage: New curriculum stage (0, 1, or 2)
        
        Returns:
            stage_changed: Whether the stage actually changed
        """
        if new_stage == self.current_curriculum_stage:
            return False
        
        old_stage = self.current_curriculum_stage
        self.current_curriculum_stage = new_stage
        
        # Update encoder wrapper's curriculum stage
        self.actor_encoder_wrapper.set_curriculum_stage(new_stage)
        
        print(f"[MAPPO] Curriculum stage changed from {old_stage} to {new_stage}")
        print("[MAPPO] Resetting optimizers for new stage...")
        
        # Reset actor optimizer
        # self._reset_actor_optimizer()
        
        # Reset critic optimizer
        # self._reset_critic_optimizer()
        
        # ===== IMPORTANT: Do NOT reset value normalizer to preserve learned statistics =====
        # This prevents value_loss explosion during curriculum transitions
        # if hasattr(self, 'value_normalizer') and self.value_normalizer is not None:
        #     self.value_normalizer.reset_parameters()
        #     print("[MAPPO] Value normalizer reset")
        print("[MAPPO] Value normalizer NOT reset (preserving learned statistics for stability)")
        
        return True
    
    def _reset_actor_optimizer(self):
        """Reset actor optimizer to initial state."""
        actor_params = list(self.actor_encoder.parameters()) + list(self.actor_body.parameters())
        
        lr = self.cfg.actor.lr
        weight_decay = getattr(self.cfg.actor, "weight_decay", 0.0)
        
        self.actor_opt = torch.optim.Adam(actor_params, lr=lr, weight_decay=weight_decay)
        
        # Re-apply learning rate scheduler if configured
        if hasattr(self.cfg.actor, "lr_scheduler") and self.cfg.actor.lr_scheduler:
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.actor.lr_scheduler)
            self.actor_scheduler = scheduler_cls(self.actor_opt, **self.cfg.actor.lr_scheduler_kwargs)
        
        print(f"[MAPPO] Actor optimizer reset with lr={lr}")
    
    def _reset_critic_optimizer(self):
        """Reset critic optimizer to initial state."""
        critic_params = list(self.critic_module.parameters())
        
        lr = self.cfg.critic.lr
        weight_decay = getattr(self.cfg.critic, "weight_decay", 0.0)
        
        self.critic_opt = torch.optim.Adam(critic_params, lr=lr, weight_decay=weight_decay)
        
        # Re-apply learning rate scheduler if configured
        if hasattr(self.cfg.critic, "lr_scheduler") and self.cfg.critic.lr_scheduler:
            scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfg.critic.lr_scheduler)
            self.critic_scheduler = scheduler_cls(self.critic_opt, **self.cfg.critic.lr_scheduler_kwargs)
        
        print(f"[MAPPO] Critic optimizer reset with lr={lr}")