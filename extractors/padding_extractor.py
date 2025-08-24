import torch
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_relative_features,
    extract_obstacle_relative_features,
    validate_observation_tensors,
)


class PaddingExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Dict, **kwargs):
        self._max_obstacles = kwargs.get("max_obstacles", 10)
        self._include_acceleration = kwargs.get("include_acceleration", False)

        self._agent_size = 4 if self._include_acceleration else 2
        self._target_size = 2
        self._obstacle_size = 6 if self._include_acceleration else 4
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
            agent_data, obstacles_data, target_data, mask, self._max_obstacles
        )

        agent_features = extract_agent_features(
            agent_data, self._include_acceleration
        )

        target_features = extract_target_relative_features(
            agent_data, target_data
        )

        obstacle_features = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, self._include_acceleration
        )

        obstacles_flat = self._flatten_obstacle_features(
            obstacle_features, self._max_obstacles, self._obstacle_size
        )

        features = torch.cat(
            [agent_features, target_features, obstacles_flat], dim=1
        )
        return features
