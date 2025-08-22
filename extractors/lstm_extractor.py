import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMExtractor(BaseFeaturesExtractor):
    """
    Converts multi-input observations into feature vectors using an LSTM-based extractor.

    Expected input (each tensor has a batch dimension [B] added by SB3):
      - obs["agent"]     : [B, A]            - [vel_x, vel_y] or [vel_x, vel_y, acc_x, acc_y]
      - obs["target"]    : [B, T]            - [pos_x, pos_y]
      - obs["obstacles"] : [B, N, F]         - [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
      - obs["mask"]      : [B, N]            - [valid_obstacle_1, ..., valid_obstacle_N]

    Output:
      - features         : [B, features_dim] - Extracted feature vector

    Notes:
      - Pass the same `features_dim` as padding_extractor in `policy_kwargs`.
      - The LSTM treats the obstacles dimension (N) as the sequence length and uses the last hidden state.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,     # Same as padding_extractor
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # Extract dimensions from observation space
        agent_dim  = int(observation_space["agent"].shape[-1])  # [vel_x, vel_y] or [vel_x, vel_y, acc_x, acc_y]
        target_dim = int(observation_space["target"].shape[-1])  # [pos_x, pos_y]

        # obstacles: [N, F]
        obs_space: spaces.Box = observation_space["obstacles"]
        assert len(obs_space.shape) == 2, "obstacles must have shape (N, F)"
        self.max_obstacles = int(obs_space.shape[0])   # N
        obs_feat_dim       = int(obs_space.shape[1])   # F

        # mask: [N]
        mask_space: spaces.Box = observation_space["mask"]
        assert len(mask_space.shape) == 1 and int(mask_space.shape[0]) == self.max_obstacles, \
            "mask must have shape (N,) and match obstacles' first dim"

        # LSTM backbone (encodes obstacle sequences)
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=obs_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Combine Agent/Target with LSTM output and project to final features_dim
        fusion_in = agent_dim + target_dim + lstm_out_dim  # [A + T + H]
        layers = [nn.Linear(fusion_in, features_dim)]
        if use_layernorm:
            layers.append(nn.LayerNorm(features_dim))
        layers.append(nn.ReLU())
        self.proj = nn.Sequential(*layers)

        # Final output dimension referenced by SB3
        self.features_dim = features_dim

    @torch.no_grad()
    def _lengths_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Derives sequence lengths from mask ∈ {0,1}, shape [B, N].
        Clamps minimum length to 1 to avoid empty sequences.
        """
        if mask.dtype.is_floating_point:
            lengths = (mask > 0.5).sum(dim=1)  # [B]
        else:
            lengths = mask.long().sum(dim=1)  # [B]
        return lengths.clamp(min=1)

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Generates feature vectors from the observation dictionary, returning [B, features_dim].
        """
        agent      = obs["agent"].float()        # [B, A] - [vel_x, vel_y] or [vel_x, vel_y, acc_x, acc_y]
        target     = obs["target"].float()       # [B, T] - [pos_x, pos_y]
        obstacles  = obs["obstacles"].float()    # [B, N, F] - [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        mask       = obs["mask"].float()         # [B, N] - [valid_obstacle_1, ..., valid_obstacle_N]

        B, N, F = obstacles.shape  # Batch size, max obstacles, obstacle features

        # Handle variable-length sequences using pack_padded_sequence
        lengths = self._lengths_from_mask(mask)                        # [B]
        packed = nn.utils.rnn.pack_padded_sequence(
            obstacles, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, c_n) = self.lstm(packed)

        # Select the last hidden state from the final LSTM layer
        if self.bidirectional:
            fwd_last = h_n[-2]  # [B, H]
            bwd_last = h_n[-1]  # [B, H]
            lstm_feat = torch.cat([fwd_last, bwd_last], dim=1)  # [B, 2H]
        else:
            lstm_feat = h_n[-1]  # [B, H]

        # Combine Agent/Target with LSTM output and project to features_dim
        fused = torch.cat([agent, target, lstm_feat], dim=1)  # [B, A+T+H]
        features = self.proj(fused)                           # [B, features_dim]
        return features
