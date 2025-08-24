import torch
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_relative_features,
    extract_obstacle_relative_features,
    flatten_obstacle_features,
    get_feature_dimensions,
    validate_observation_tensors,
)


class PaddingExtractor(BaseFeaturesExtractor):
    """
    A simple feature extractor that concatenates all features into a single vector.

    This extractor implements the most straightforward approach to feature extraction
    by flattening all obstacle features and concatenating them with agent and target
    features. It's called "padding" because it handles variable numbers of obstacles
    by padding shorter sequences to a fixed maximum length.

    The extracted features are organized as:
    [agent_features, target_features, obstacle1_features, obstacle2_features, ...]

    This approach is simple and works well when the number of obstacles is relatively
    small and consistent, but may not scale well to environments with many obstacles
    or highly variable obstacle counts.

    Args:
        observation_space: The observation space from the Gymnasium environment
        max_obstacles: Maximum number of obstacles that can be present (default: 10)
        include_acceleration: Whether to include acceleration in features (default: False)
    """

    def __init__(self, observation_space: spaces.Dict, **kwargs):
        self._max_obstacles = kwargs.get("max_obstacles", 10)
        self._include_acceleration = kwargs.get("include_acceleration", False)

        (
            self._agent_size,
            self._target_size,
            self._obstacle_size,
            self._obstacles_total_size,
        ) = get_feature_dimensions(
            self._max_obstacles, self._include_acceleration
        )

        self._features_dim = (
            self._agent_size + self._target_size + self._obstacles_total_size
        )

        super().__init__(observation_space, self._features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and concatenate all features into a single vector.

        This method processes the input observations by extracting agent-relative
        features and concatenating them into a single feature vector. All obstacles
        are flattened into a single sequence, making this suitable for fully
        connected layers but not for attention or sequence-based processing.

        Args:
            observations: Dictionary containing:
                - "agent": Tensor [batch_size, 7] with agent state
                - "obstacles": Tensor [batch_size, max_obstacles, 7] with obstacle states
                - "target": Tensor [batch_size, 2] with target position
                - "mask": Tensor [batch_size, max_obstacles] with obstacle validity

        Returns:
            Tensor of shape [batch_size, features_dim] containing the concatenated
            features: [agent_features, target_features, flattened_obstacle_features]
        """
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

        obstacles_flat = flatten_obstacle_features(
            obstacle_features, self._max_obstacles, self._obstacle_size
        )

        features = torch.cat(
            [agent_features, target_features, obstacles_flat], dim=1
        )
        return features
