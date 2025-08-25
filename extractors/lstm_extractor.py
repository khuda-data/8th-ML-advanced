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


class LSTMExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        max_obstacles: int = 10,
        include_acceleration: bool = True,
        include_radius: bool = True,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        use_layernorm: bool = True,
        features_dim: int = 64,
        **kwargs,
    ) -> None:
        self._max_obstacles = max_obstacles
        self._include_acceleration = include_acceleration
        self._include_radius = include_radius
        self._bidirectional = bidirectional

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

        super().__init__(observation_space, features_dim)

        self._lstm = nn.LSTM(
            input_size=self._obstacle_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        total_input_dim = self._agent_size + self._target_size + lstm_out_dim

        if use_layernorm:
            self._post = nn.Sequential(
                nn.Linear(total_input_dim, features_dim * 2),
                nn.LayerNorm(features_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(features_dim * 2, features_dim),
            )
        else:
            self._post = nn.Sequential(
                nn.Linear(total_input_dim, features_dim * 2),
                nn.LeakyReLU(),
                nn.Linear(features_dim * 2, features_dim),
            )

    def _compute_sequence_lengths_from_mask(
        self, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.dtype.is_floating_point:
            lengths = (mask > 0.5).sum(dim=1)
        else:
            lengths = mask.long().sum(dim=1)

        return lengths.clamp(min=1)

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
        )

        agent_features = extract_agent_features(
            agent_data, self._include_acceleration, self._include_radius
        )

        target_features = extract_target_features(target_data)

        rel_feats = extract_obstacle_features(
            obstacles_data,
            mask,
            self._include_acceleration,
            self._include_radius,
        )

        lengths = self._compute_sequence_lengths_from_mask(mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            rel_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self._lstm(packed)

        if self._bidirectional:
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            lstm_features = torch.cat([forward_hidden, backward_hidden], dim=-1)
        else:
            lstm_features = hidden[-1]

        features = torch.cat(
            [agent_features, target_features, lstm_features], dim=1
        )

        return self._post(features)
