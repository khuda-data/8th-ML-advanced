import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_relative_features,
    extract_obstacle_relative_features_vectorized,
    flatten_obstacle_features,
    get_feature_dimensions,
    validate_observation_tensors,
    compute_sequence_lengths_from_mask,
)


class LSTMExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using LSTM to process obstacle sequences.

    This extractor uses LSTM (Long Short-Term Memory) networks to process
    obstacle features as sequences, allowing the model to capture temporal
    or spatial dependencies between obstacles. Unlike attention mechanisms
    that process all obstacles simultaneously, LSTM processes obstacles
    sequentially, which can be beneficial for capturing ordering relationships.

    The LSTM processes relative obstacle features and outputs a processed
    representation for each obstacle, which is then flattened and concatenated
    with agent and target features. This approach is particularly useful when:
    - Obstacle ordering matters (e.g., obstacles encountered in sequence)
    - Sequential dependencies between obstacles exist
    - A smaller model size is preferred compared to attention

    Architecture:
    1. Extract relative obstacle features
    2. Pack sequences to handle variable obstacle counts efficiently
    3. Process through LSTM layers with optional bidirectionality
    4. Unpack and project back to original feature space
    5. Flatten and concatenate with agent/target features

    Args:
        observation_space: The observation space from the Gymnasium environment
        max_obstacles: Maximum number of obstacles (default: 10)
        include_acceleration: Whether to include acceleration features (default: False)
        lstm_hidden: Hidden size of LSTM layers (default: 128)
        lstm_layers: Number of LSTM layers (default: 1)
        bidirectional: Whether to use bidirectional LSTM (default: False)
        use_layernorm: Whether to apply layer normalization (default: True)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        max_obstacles: int = 10,
        include_acceleration: bool = False,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        use_layernorm: bool = True,
    ) -> None:
        self._max_obstacles = max_obstacles
        self._include_acceleration = include_acceleration

        # Get standard feature dimensions
        (
            self._agent_size,
            self._target_size,
            self._obstacle_size,
            self._obstacles_total_size,
        ) = get_feature_dimensions(max_obstacles, include_acceleration)

        out_dim = (
            self._agent_size + self._target_size + self._obstacles_total_size
        )

        super().__init__(observation_space, out_dim)

        # LSTM processes per-obstacle relative features [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, (acc_x, acc_y)]
        self._lstm = nn.LSTM(
            input_size=self._obstacle_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Project LSTM output back to per-obstacle feature size to maintain compatibility
        self._per_step_head = nn.Linear(lstm_out_dim, self._obstacle_size)

        # Optional post-processing normalization
        self._post = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self._features_dim = out_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features using LSTM processing of obstacle sequences.

        This method processes observations by treating obstacles as a sequence
        and applying LSTM to capture dependencies between them. The LSTM output
        is projected back to obstacle feature space and then concatenated with
        agent and target features for final processing.

        The key advantage of this approach is that it can capture sequential
        relationships between obstacles while handling variable sequence lengths
        efficiently through packed sequences.

        Args:
            observations: Dictionary containing:
                - "agent": Tensor [batch_size, 7] with agent state
                - "obstacles": Tensor [batch_size, max_obstacles, 7] with obstacle states
                - "target": Tensor [batch_size, 2] with target position
                - "mask": Tensor [batch_size, max_obstacles] with obstacle validity

        Returns:
            Tensor of shape [batch_size, out_dim] containing concatenated features:
            [agent_features, target_features, lstm_processed_obstacle_features]
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

        rel_feats = extract_obstacle_relative_features_vectorized(
            agent_data, obstacles_data, mask, self._include_acceleration
        )

        lengths = compute_sequence_lengths_from_mask(mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            rel_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self._lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=self._max_obstacles
        )

        per_step = self._per_step_head(lstm_out)
        valid_mask = (mask > 0.5).float().unsqueeze(-1)
        per_step = per_step * valid_mask

        obstacles_flat = flatten_obstacle_features(
            per_step, self._max_obstacles, self._obstacle_size
        )

        features = torch.cat(
            [agent_features, target_features, obstacles_flat], dim=1
        )

        return self._post(features)
