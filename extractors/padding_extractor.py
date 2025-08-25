import torch
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_features,
    extract_obstacle_features,
    validate_observation_tensors,
)


class PaddingExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        max_obstacles=10,
        include_acceleration=True,
        include_radius=True,
        **kwargs,
    ):
        self._max_obstacles = max_obstacles
        self._include_acceleration = include_acceleration
        self._include_radius = include_radius

        agent_features = 2
        if self._include_radius:
            agent_features += 1
        if self._include_acceleration:
            agent_features += 2
        self._agent_size = agent_features

        self._target_size = 2  # rel_pos_x, rel_pos_y (always same)

        obstacle_features = 4
        if self._include_radius:
            obstacle_features += 1
        if self._include_acceleration:
            obstacle_features += 2
        self._obstacle_size = obstacle_features
        self._obstacles_total_size = self._obstacle_size * self._max_obstacles

        self._features_dim = (
            self._agent_size + self._target_size + self._obstacles_total_size
        )

        super().__init__(observation_space, self._features_dim)

    def _flatten_obstacle_features(
        self,
        obstacle_features: torch.Tensor,
        max_obstacles: int,
        feature_size: int,
    ) -> torch.Tensor:
        batch_size = obstacle_features.shape[0]
        return obstacle_features.reshape(
            batch_size, max_obstacles * feature_size
        )

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
            self._include_radius,
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

        obstacles_flat = self._flatten_obstacle_features(
            obstacle_features, self._max_obstacles, self._obstacle_size
        )

        features = torch.cat(
            [agent_features, target_features, obstacles_flat], dim=1
        )
        return features
