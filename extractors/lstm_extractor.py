import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor compatible with SB3.
    Input keys and semantics follow padding_extractor.py:
      - observations["agent"]: [batch_size, 7] - [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
      - observations["obstacles"]: [batch_size, max_obstacles, 7] - each obstacle [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
      - observations["target"]: [batch_size, 2] - [pos_x, pos_y]
      - observations["mask"]: [batch_size, max_obstacles] - 1 if obstacle valid, else 0

    Internally computes relative obstacle features (agent-centered):
      - without acceleration: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
      - with acceleration: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]

    Output shape matches padding_extractor.py:
      [batch_size, agent_features + target_features + max_obstacles * obstacle_feature_size]
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
        # same feature sizing rules as padding_extractor
        self.max_obstacles = max_obstacles
        self.include_acceleration = include_acceleration
        self._agent_size = 4 if include_acceleration else 2   # [vel_x, vel_y] or [vel_x, vel_y, acc_x, acc_y]
        self._target_size = 2                                 # [pos_x, pos_y]
        self._obstacle_size = 6 if include_acceleration else 4
        self._obstacles_total_size = self._obstacle_size * max_obstacles
        out_dim = self._agent_size + self._target_size + self._obstacles_total_size

        super().__init__(observation_space, out_dim)

        # LSTM processes per-obstacle relative features [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, (acc_x, acc_y)]
        self.lstm = nn.LSTM(
            input_size=self._obstacle_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Project LSTM output back to per-obstacle feature size to maintain compatibility
        self.per_step_head = nn.Linear(lstm_out_dim, self._obstacle_size)

        # Optional post-processing normalization
        self.post = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.features_dim = out_dim

    @torch.no_grad()
    def _lengths_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute valid sequence lengths from mask [batch_size, max_obstacles].
        Ensures minimum length of 1 to avoid empty sequence issues.
        """
        if mask.dtype.is_floating_point:
            lengths = (mask > 0.5).sum(dim=1)
        else:
            lengths = mask.long().sum(dim=1)
        return lengths.clamp(min=1)

    def _build_relative_obstacle_feats(
        self,
        agent_data: torch.Tensor,       # [batch_size, 7] - [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        obstacles_data: torch.Tensor,   # [batch_size, max_obstacles, 7]
        mask: torch.Tensor,             # [batch_size, max_obstacles]
    ) -> torch.Tensor:
        """
        Compute per-obstacle relative features with respect to agent.
        Returns [batch_size, max_obstacles, obstacle_feature_size], masked where invalid.
        """
        agent_pos = agent_data[:, [1, 2]].unsqueeze(1)   # [pos_x, pos_y]
        agent_vel = agent_data[:, [3, 4]].unsqueeze(1)   # [vel_x, vel_y]
        obs_pos = obstacles_data[:, :, [1, 2]]           # [pos_x, pos_y]
        obs_vel = obstacles_data[:, :, [3, 4]]           # [vel_x, vel_y]

        rel_pos = obs_pos - agent_pos                    # [rel_pos_x, rel_pos_y]
        rel_vel = obs_vel - agent_vel                    # [rel_vel_x, rel_vel_y]

        if self.include_acceleration:
            obs_acc = obstacles_data[:, :, [5, 6]]       # [acc_x, acc_y]
            feats = torch.cat([rel_pos, rel_vel, obs_acc], dim=-1)
        else:
            feats = torch.cat([rel_pos, rel_vel], dim=-1)

        valid_mask = (mask > 0.5).float().unsqueeze(-1)
        return feats * valid_mask

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass returning [batch_size, features_dim].
        """
        agent_data = observations["agent"]         # [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        obstacles_data = observations["obstacles"] # [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y] per obstacle
        target_data = observations["target"]       # [pos_x, pos_y]
        mask = observations["mask"]                # obstacle validity mask

        # agent velocity (+acceleration if enabled)
        if self.include_acceleration:
            agent_features = agent_data[:, [3, 4, 5, 6]]  # [vel_x, vel_y, acc_x, acc_y]
        else:
            agent_features = agent_data[:, [3, 4]]        # [vel_x, vel_y]

        target_features = target_data                     # [pos_x, pos_y]

        # relative obstacle features [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y,(acc_x, acc_y)]
        rel_feats = self._build_relative_obstacle_feats(agent_data, obstacles_data, mask)

        # pack valid sequences for LSTM
        lengths = self._lengths_from_mask(mask)
        packed = nn.utils.rnn.pack_padded_sequence(
            rel_feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=self.max_obstacles
        )

        # project each obstacle timestep back to obstacle_feature_size
        per_step = self.per_step_head(lstm_out)
        valid_mask = (mask > 0.5).float().unsqueeze(-1)
        per_step = per_step * valid_mask

        # flatten obstacle features
        obstacles_flat = per_step.reshape(agent_data.shape[0], self._obstacles_total_size)

        # concatenate agent, target, and obstacle features
        features = torch.cat([agent_features, target_features, obstacles_flat], dim=1)

        return self.post(features)
