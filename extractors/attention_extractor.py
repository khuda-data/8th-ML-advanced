import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_features,
    extract_obstacle_features,
    validate_observation_tensors,
)


class AttentionExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        max_obstacles: int = 10,
        include_acceleration: bool = False,
        include_radius: bool = True,
        **kwargs,
    ):
        self._d_model = d_model
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._max_obstacles = max_obstacles
        self._include_acceleration = include_acceleration
        self._include_radius = include_radius

        agent_features = 2
        if include_radius:
            agent_features += 1
        if include_acceleration:
            agent_features += 2
        self._agent_size = agent_features

        self._target_size = 2  # rel_pos_x, rel_pos_y (always same)

        obstacle_features = 4
        if include_radius:
            obstacle_features += 1
        if include_acceleration:
            obstacle_features += 2
        self._obstacle_size = obstacle_features

        features_dim = self._target_size + d_model
        super().__init__(observation_space, features_dim)

        self._initial_agent_projection = nn.Linear(self._agent_size, d_model)

        self._q_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self._k_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )
        self._v_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )

        self._mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=num_heads, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        self._layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        self._feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.LeakyReLU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self._output_projection = nn.Linear(d_model, d_model)

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return mask == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        agent_data = observations["agent"]
        obstacles_data = observations["obstacles"]
        target_data = observations["target"]
        mask = observations["mask"]

        validate_observation_tensors(
            agent_data,
            obstacles_data,
            target_data,
            mask,
            self._max_obstacles,
            self._include_acceleration,
        )

        agent_features = extract_agent_features(
            agent_data, self._include_acceleration, self._include_radius
        )

        target_features = extract_target_features(target_data)

        obstacle_features = extract_obstacle_features(
            obstacles_data,
            mask,
            self._include_acceleration,
            self._include_radius,
        )

        attended_features = self._apply_attention(
            agent_features, obstacle_features, mask
        )

        return torch.cat([target_features, attended_features], dim=1)

    def _apply_attention(
        self,
        agent_features: torch.Tensor,
        obstacle_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        attended_output = self._initial_agent_projection(
            agent_features
        ).unsqueeze(1)

        layers = zip(
            range(self._num_layers),
            self._q_projections,
            self._k_projections,
            self._v_projections,
            self._mha_layers,
            self._layer_norms,
            self._feed_forwards,
        )

        for _, q_proj, k_proj, v_proj, mha, ln, ff in layers:
            q = q_proj(attended_output.squeeze(1)).unsqueeze(1)
            k = k_proj(obstacle_features)
            v = v_proj(obstacle_features)

            residual = attended_output

            attention_mask = self._create_attention_mask(mask)

            mha_output, _ = mha(
                query=q,
                key=k,
                value=v,
                key_padding_mask=attention_mask,
                need_weights=False,
            )

            attended_output = ln(residual + mha_output)

            residual = attended_output
            ff_output = ff(attended_output)
            attended_output = residual + ff_output

        attended_output = attended_output.squeeze(1)
        return self._output_projection(attended_output)
