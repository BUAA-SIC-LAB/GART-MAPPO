# MIT License
#
# Copyright (c) 2025
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

"""Graph-Recurrent-Attention encoder blocks used by MAPPO variants.

These modules build the ego-centric graph encoder, the task-context attention
module, and the temporal GRU fusion required by the Graph-Attention MAPPO
policy. They operate on agent-level TensorDict observations without assuming
PyG availability to keep the dependency footprint minimal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from tensordict import TensorDict
from torchrl.data import CompositeSpec
from .networks import MLP


@dataclass
class GraphRecurrentAttentionConfig:
    """Hyperparameters for the graph-recurrent-attention encoder."""

    embed_dim: int = 128
    gnn_heads: int = 4
    gnn_layers: int = 2
    gnn_dropout: float = 0.0
    task_heads: int = 4
    fusion_hidden_dim: int = 256
    gru_hidden_dim: int = 256
    post_mlp_units: Sequence[int] = (256, 128)
    
    # Ablation flags
    use_graph: bool = True
    use_task_attn: bool = True
    use_gru: bool = True


class EgoGraphAttention(nn.Module):
    """Applies ego-centric graph attention over self and neighbour features."""

    def __init__(self, self_dim: int, other_dim: int, cfg: GraphRecurrentAttentionConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.self_proj = nn.Linear(self_dim, self.embed_dim)
        self.other_proj = nn.Linear(other_dim, self.embed_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(cfg.gnn_dropout)
        for _ in range(cfg.gnn_layers):
            self.layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=cfg.gnn_heads,
                    batch_first=True,
                )
            )
            self.norms.append(nn.LayerNorm(self.embed_dim))

    def forward(self, self_feat: torch.Tensor, others_feat: torch.Tensor) -> torch.Tensor:
        """Runs attention message passing.

        Args:
            self_feat: [B, N, self_dim] tensor containing the ego state.
            others_feat: [B, N, N-1, other_dim] tensor of neighbour features.

        Returns:
            [B * N, embed_dim] tensor for the ego node after message passing.
        """
        if self_feat.ndim != 3 or others_feat.ndim != 4:
            raise RuntimeError("Unexpected feature shapes for ego graph encoder")

        num_envs, num_agents, _ = self_feat.shape
        self_tokens = self.self_proj(self_feat)
        other_tokens = self.other_proj(others_feat)
        node_tokens = torch.cat([self_tokens.unsqueeze(-2), other_tokens], dim=-2)
        node_tokens = node_tokens.reshape(-1, num_agents, self.embed_dim)

        for attn, norm in zip(self.layers, self.norms):
            residual = node_tokens
            attn_out, _ = attn(node_tokens, node_tokens, node_tokens)
            node_tokens = norm(residual + self.dropout(attn_out))

        return node_tokens[:, 0]


class TaskContextAttention(nn.Module):
    """Aggregates task-specific tokens using cross-attention."""

    def __init__(self, gate_dim: int, formation_dim: int, endpoint_dim: int, cfg: GraphRecurrentAttentionConfig):
        super().__init__()
        self.embed_dim = cfg.embed_dim
        self.gate_proj = nn.Linear(gate_dim, self.embed_dim)
        self.formation_proj = nn.Linear(formation_dim, self.embed_dim)
        self.endpoint_proj = nn.Linear(endpoint_dim, self.embed_dim)
        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=cfg.task_heads,
            batch_first=True,
        )
        self.norm_tokens = nn.LayerNorm(self.embed_dim)
        self.norm_query = nn.LayerNorm(self.embed_dim)

    def forward(
        self, 
        agent_embedding: torch.Tensor, 
        gate: torch.Tensor, 
        formation: torch.Tensor, 
        endpoint: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Produces a task-aware embedding aligned with the agent query.
        
        Args:
            agent_embedding: [batch_size, embed_dim] agent features
            gate: [batch_size, gate_dim] gate information
            formation: [batch_size, formation_dim] formation target
            endpoint: [batch_size, endpoint_dim] endpoint information
            attn_mask: [batch_size, 3] boolean mask where False means attend, True means mask out
                       Order: [gate_mask, formation_mask, endpoint_mask]
                       
        Returns:
            context: [batch_size, embed_dim] task-aware embedding
        """
        gate_token = self.norm_tokens(self.gate_proj(gate))
        formation_token = self.norm_tokens(self.formation_proj(formation))
        endpoint_token = self.norm_tokens(self.endpoint_proj(endpoint))
        tokens = torch.stack([gate_token, formation_token, endpoint_token], dim=1)  # [B, 3, embed_dim]

        query = self.norm_query(self.query_proj(agent_embedding)).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Apply attention mask if provided
        # PyTorch MultiheadAttention expects mask shape [batch_size * num_heads, tgt_len, src_len]
        # or [tgt_len, src_len] for batch_first=True with broadcasting
        # attn_mask values: True (or positive inf) masks out that position
        key_padding_mask = None
        if attn_mask is not None:
            # attn_mask is [B, 3] where True means mask out
            # For MultiheadAttention with batch_first=True, we use key_padding_mask [B, S]
            key_padding_mask = attn_mask  # [B, 3]
        
        context, _ = self.attn(query, tokens, tokens, key_padding_mask=key_padding_mask)
        return context.squeeze(1)


class GraphRecurrentAttentionEncoder(nn.Module):
    """Combines ego-centric graph attention, task attention, and a GRU cell."""

    def __init__(self, observation_spec: CompositeSpec, cfg: GraphRecurrentAttentionConfig):
        super().__init__()
        # Handle potential extra dimensions in observation specs
        obs_self_spec = observation_spec["obs_self"]
        obs_others_spec = observation_spec["obs_others"]
        
        # obs_self shape: (1, obs_self_dim) or (obs_self_dim,)
        obs_self_dim = obs_self_spec.shape[-1]
        # obs_others shape: (num_agents-1, obs_other_dim)
        obs_other_dim = obs_others_spec.shape[-1]
        
        gate_dim = observation_spec["gate_info"].shape[-1]
        formation_dim = observation_spec["formation_target"].shape[-1]
        endpoint_dim = observation_spec["endpoint_info"].shape[-1]

        # Calculate num_agents from obs_others shape
        # obs_others has shape (num_agents-1, obs_dim)
        self.num_agents = obs_others_spec.shape[-2] + 1
        self.cfg = cfg

        if cfg.use_graph:
            self.graph = EgoGraphAttention(obs_self_dim, obs_other_dim, cfg)
        else:
            # Ablation: MLP instead of Graph Attention
            # Input: self + flattened others
            mlp_input_dim = obs_self_dim + (self.num_agents - 1) * obs_other_dim
            self.graph_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, cfg.embed_dim),
                # nn.LayerNorm(cfg.embed_dim),
                # nn.Mish()
            )

        if cfg.use_task_attn:
            self.task = TaskContextAttention(gate_dim, formation_dim, endpoint_dim, cfg)
        else:
            # Ablation: MLP instead of Task Attention
            # Input: gate + formation + endpoint
            mlp_input_dim = gate_dim + formation_dim + endpoint_dim
            self.task_mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, cfg.embed_dim),
                # nn.LayerNorm(cfg.embed_dim),
                # nn.Mish()
            )

        fusion_input = cfg.embed_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, cfg.fusion_hidden_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.fusion_hidden_dim),
        )
        
        if cfg.use_gru:
            self.gru = nn.GRUCell(cfg.fusion_hidden_dim, cfg.gru_hidden_dim)
            post_input = cfg.fusion_hidden_dim + cfg.gru_hidden_dim
        else:
            # Ablation: No GRU
            post_input = cfg.fusion_hidden_dim

        post_units = list(cfg.post_mlp_units)
        if not post_units:
            raise ValueError("post_mlp_units must contain at least one element")
        self.post_mlp = nn.Sequential(
            nn.LayerNorm(post_input),
            MLP([post_input, *post_units]),
        )
        self.output_dim = post_units[-1]

    def forward(
        self,
        observation: TensorDict,
        is_init: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
        curriculum_stage: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Runs the encoder and returns features alongside the updated hidden state.
        
        Args:
            curriculum_stage: Current curriculum learning stage (0, 1, or 2)
                0: Endpoint only - mask formation and gate attention
                1: Formation - mask gate attention only
                2: Full task - no masking
        """

        obs_self = observation.get("obs_self")
        obs_others = observation.get("obs_others")
        gate_info = observation.get("gate_info")
        formation_target = observation.get("formation_target")
        endpoint_info = observation.get("endpoint_info")

        # Handle the extra singleton dimension from the environment
        # The environment returns shapes like [batch, num_agents, 1, obs_dim]
        # We need [batch, num_agents, obs_dim]
        if obs_self.ndim == 4:
            obs_self = obs_self.squeeze(-2)  # Remove the singleton dimension
        if gate_info.ndim == 4:
            gate_info = gate_info.squeeze(-2)
        if formation_target.ndim == 4:
            formation_target = formation_target.squeeze(-2)
        if endpoint_info.ndim == 4:
            endpoint_info = endpoint_info.squeeze(-2)
        
        if obs_self.ndim < 3:
            raise RuntimeError("Expected obs_self to contain per-agent features")

        batch_size = obs_self.shape[0]
        num_agents = self.num_agents
        
        # Ensure correct shapes
        obs_self = obs_self.reshape(batch_size, num_agents, -1)
        obs_others = obs_others.reshape(batch_size, num_agents, num_agents - 1, -1)
        gate_info = gate_info.reshape(batch_size, num_agents, -1)
        formation_target = formation_target.reshape(batch_size, num_agents, -1)
        endpoint_info = endpoint_info.reshape(batch_size, num_agents, -1)

        flat_size = batch_size * num_agents

        # ===== Curriculum Learning: Create Attention Mask =====
        # Create a boolean mask [flat_size, 3] for [gate, formation, endpoint]
        # True = mask out (don't attend), False = attend to this token
        attn_mask = None
        # if curriculum_stage is not None:
        #     attn_mask = torch.zeros(flat_size, 3, dtype=torch.bool, device=obs_self.device)
        #     if curriculum_stage == 0:
        #         # Stage 0: Endpoint only - mask gate (index 0) and formation (index 1)
        #         attn_mask[:, 0] = True  # mask gate
        #         attn_mask[:, 1] = True  # mask formation
        #         # attn_mask[:, 2] remains False (attend to endpoint)
        #     elif curriculum_stage == 1:
        #         # Stage 1: Formation + Endpoint - mask gate only
        #         attn_mask[:, 0] = True  # mask gate
        #         # attn_mask[:, 1] remains False (attend to formation)
        #         # attn_mask[:, 2] remains False (attend to endpoint)
        #     # Stage 2 or None: No masking (all False = attend to all)

        if self.cfg.use_graph:
            agent_embedding = self.graph(obs_self, obs_others)
        else:
            # Flatten others: [B, N, N-1, D] -> [B*N, (N-1)*D]
            flat_others = obs_others.reshape(flat_size, -1)
            flat_self = obs_self.reshape(flat_size, -1)
            flat_input = torch.cat([flat_self, flat_others], dim=-1)
            agent_embedding = self.graph_mlp(flat_input)

        gate_token = gate_info.reshape(flat_size, -1)
        formation_token = formation_target.reshape(flat_size, -1)
        endpoint_token = endpoint_info.reshape(flat_size, -1)
        
        if self.cfg.use_task_attn:
            # Pass attention mask to task attention module
            task_embedding = self.task(
                agent_embedding, 
                gate_token, 
                formation_token, 
                endpoint_token,
                attn_mask=attn_mask
            )
        else:
            task_input = torch.cat([gate_token, formation_token, endpoint_token], dim=-1)
            task_embedding = self.task_mlp(task_input)

        fusion = self.fusion_mlp(torch.cat([agent_embedding, task_embedding], dim=-1))

        if self.cfg.use_gru:
            if hidden_state is None or hidden_state.numel() == 0:
                hidden_state = fusion.new_zeros(flat_size, self.cfg.gru_hidden_dim)
            else:
                hidden_state = hidden_state.reshape(flat_size, self.cfg.gru_hidden_dim)

            if is_init is not None:
                reset = self._prepare_is_init(is_init, batch_size, num_agents)
                hidden_state = hidden_state * (1.0 - reset)

            updated_hidden = self.gru(fusion, hidden_state)
            combined = torch.cat([fusion, updated_hidden], dim=-1)
        else:
            # No GRU
            if hidden_state is None or hidden_state.numel() == 0:
                 updated_hidden = fusion.new_zeros(flat_size, self.cfg.gru_hidden_dim) # Dummy hidden
            else:
                 updated_hidden = hidden_state # Pass through
            
            combined = fusion # Just fusion features
        features = self.post_mlp(combined)

        features = features.view(batch_size, num_agents, -1)
        updated_hidden = updated_hidden.view(batch_size, num_agents, -1)
        return features, updated_hidden

    @staticmethod
    def _prepare_is_init(is_init: torch.Tensor, batch: int, agents: int) -> torch.Tensor:
        reset = is_init
        if reset.dim() == 1:
            reset = reset.unsqueeze(-1)
        while reset.dim() < 3:
            reset = reset.unsqueeze(-2)
        if reset.shape[-2] == 1 and agents > 1:
            reset = reset.expand(batch, agents, 1)
        reset = reset.reshape(batch * agents, 1).to(dtype=torch.float32)
        return reset


class CentralValueAggregator(nn.Module):
    """Aggregates per-agent features with central observations for critic value."""

    def __init__(self, feature_dim: int, central_spec: CompositeSpec, hidden_units: Sequence[int]):
        super().__init__()
        drones_dim = central_spec["drones"].shape[-1]
        gates_dim = central_spec["gates"].shape[-1]
        formation_dim = central_spec["formation"].shape[-1]


        self.drone_proj = nn.Sequential(
            nn.Linear(drones_dim, feature_dim),
            nn.Mish(),
        )
        self.gate_proj = nn.Sequential(
            nn.Linear(gates_dim, feature_dim),
            nn.Mish(),
        )
        self.formation_proj = nn.Sequential(
            nn.Linear(formation_dim, feature_dim),
            nn.Mish(),
        )

        input_dim = feature_dim * 4
        layers = [input_dim, *hidden_units]
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLP(layers),
            nn.Mish(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(layers[-1], 1),
            nn.Mish(),
        )

    def forward(self, features: torch.Tensor, central_obs: TensorDict) -> torch.Tensor:
        """
        Args:
            features: Latent features, shape can be:
                - [B, N, D] during data collection
                - [B, T, N, D] during training (with time dimension)
            central_obs: Central observations
        
        Returns:
            values: shape [B, N, 1] or [B, T, N, 1]
        """
        drones = central_obs.get("drones")
        gates = central_obs.get("gates")
        formation = central_obs.get("formation")

        # Project and aggregate drone features
        drones_ctx = self.drone_proj(drones).mean(dim=-2)  # [..., feature_dim]
        
        # Handle gates (may have variable number)
        if gates.shape[-2] > 0:
            gates_ctx = self.gate_proj(gates).mean(dim=-2)  # [..., feature_dim]
        else:
            raise RuntimeError("No gate information available in central observations")
        
        # Project formation
        formation_ctx = self.formation_proj(formation).mean(dim=-2)  # [..., feature_dim]

        # Concatenate all context
        global_ctx = torch.cat([drones_ctx, gates_ctx, formation_ctx], dim=-1)  # [..., feature_dim*3]
        
        # Broadcast global context to match features shape
        # features can be [B, N, D] or [B, T, N, D]
        # global_ctx is [..., feature_dim*3] where ... matches features except N dimension
        
        # Determine the agent dimension position
        if features.ndim == 3:
            # Data collection: [B, N, D]
            num_agents = features.size(1)
            global_ctx = global_ctx.unsqueeze(1).expand(-1, num_agents, -1)  # [B, N, feature_dim*3]
        elif features.ndim == 4:
            # Training with time: [B, T, N, D]
            num_agents = features.size(2)
            # global_ctx shape: [B, T, feature_dim*3]
            global_ctx = global_ctx.unsqueeze(2).expand(-1, -1, num_agents, -1)  # [B, T, N, feature_dim*3]
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")

        # Concatenate features with global context
        critic_input = torch.cat([features, global_ctx], dim=-1)  # [..., N, D + feature_dim*3]
        
        # Process through MLP and value head
        critic_feat = self.mlp(critic_input)  # [..., N, hidden_dim]
        values = self.value_head(critic_feat)  # [..., N, 1]
        
        return values