import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        max_obstacles: int = 10,
        include_acceleration: bool = False,
        **kwargs,
    ):
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_obstacles = max_obstacles
        self.include_acceleration = include_acceleration

        self._agent_size = 4 if include_acceleration else 2
        self._target_size = 2
        self._obstacle_size = 5 if include_acceleration else 4

        features_dim = self._target_size + d_model
        super().__init__(observation_space, features_dim)

        self.initial_agent_projection = nn.Linear(self._agent_size, d_model)

        self.q_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self.k_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )
        self.v_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )

        self.mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=num_heads, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        self.feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        agent_data = observations["agent"]  # [B, 7]
        obstacles_data = observations["obstacles"]  # [B, max_obstacles, 7]
        target_data = observations["target"]  # [B, 2]
        mask = observations["mask"]  # [B, max_obstacles]

        agent_features = self._extract_agent_features(
            agent_data
        )  # [B, agent_size]
        target_features = target_data  # [B, 2]
        obstacle_features = self._extract_obstacle_features(
            agent_data, obstacles_data, mask
        )  # [B, max_obstacles, obstacle_size]

        attended_features = self._apply_attention(
            agent_features, obstacle_features, mask
        )  # [B, d_model]

        return torch.cat(
            [target_features, attended_features], dim=1
        )  # [B, 2 + d_model]

    def _extract_agent_features(self, agent_data: torch.Tensor) -> torch.Tensor:
        # [B, 7] -> [B, 2] or [B, 4]
        if self.include_acceleration:
            return agent_data[:, [3, 4, 5, 6]]  # vel + acc
        else:
            return agent_data[:, [3, 4]]  # vel only

    def _extract_obstacle_features(
        self,
        agent_data: torch.Tensor,  # [B, 7]
        obstacles_data: torch.Tensor,  # [B, max_obstacles, 7]
        mask: torch.Tensor,  # [B, max_obstacles]
    ) -> torch.Tensor:
        batch_size = agent_data.shape[0]
        output = torch.zeros(
            batch_size,
            self.max_obstacles,
            self._obstacle_size,
            device=agent_data.device,
        )  # [B, max_obstacles, obstacle_size]

        agent_pos = agent_data[:, [1, 2]]  # [B, 2]
        agent_vel = agent_data[:, [3, 4]]  # [B, 2]

        for i in range(self.max_obstacles):
            obstacle = obstacles_data[:, i]  # [B, 7]
            valid_mask = mask[:, i] > 0  # [B]

            if valid_mask.any():
                obstacle_pos = obstacle[:, [1, 2]]  # [B, 2]
                obstacle_vel = obstacle[:, [3, 4]]  # [B, 2]
                rel_pos = obstacle_pos - agent_pos  # [B, 2]
                rel_vel = obstacle_vel - agent_vel  # [B, 2]

                if self.include_acceleration:
                    obstacle_acc = obstacle[:, [5, 6]]  # [B, 2]
                    features = torch.cat(
                        [rel_pos, rel_vel, obstacle_acc], dim=1
                    )  # [B, 5]
                else:
                    features = torch.cat([rel_pos, rel_vel], dim=1)  # [B, 4]

                output[:, i] = features * valid_mask.unsqueeze(
                    1
                )  # [B, obstacle_size]

        return output  # [B, max_obstacles, obstacle_size]

    def _apply_attention(
        self,
        agent_features: torch.Tensor,  # [B, agent_size]
        obstacle_features: torch.Tensor,  # [B, max_obstacles, obstacle_size]
        mask: torch.Tensor,  # [B, max_obstacles]
    ) -> torch.Tensor:
        attended_output = self.initial_agent_projection(
            agent_features
        ).unsqueeze(1)
        # [B, 1, d_model]

        layers = zip(
            range(self.num_layers),
            self.q_projections,
            self.k_projections,
            self.v_projections,
            self.mha_layers,
            self.layer_norms,
            self.feed_forwards,
        )

        for i, q_proj, k_proj, v_proj, mha, ln, ff in layers:
            q = q_proj(attended_output.squeeze(1)).unsqueeze(
                1
            )  # [B, 1, d_model]
            k = k_proj(obstacle_features)  # [B, max_obstacles, d_model]
            v = v_proj(obstacle_features)  # [B, max_obstacles, d_model]

            residual = attended_output  # [B, 1, d_model]

            mha_output, _ = mha(
                query=q,  # [B, 1, d_model]
                key=k,  # [B, max_obstacles, d_model]
                value=v,  # [B, max_obstacles, d_model]
                key_padding_mask=mask == 0,
                # [B, max_obstacles] - True for invalid
                need_weights=False,
            )  # -> [B, 1, d_model]

            attended_output = ln(residual + mha_output)  # [B, 1, d_model]

            residual = attended_output
            ff_output = ff(attended_output)  # [B, 1, d_model]
            attended_output = residual + ff_output  # [B, 1, d_model]

        attended_output = attended_output.squeeze(1)  # [B, d_model]
        return self.output_projection(attended_output)  # [B, d_model]
