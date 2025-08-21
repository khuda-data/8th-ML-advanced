import torch
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PaddingExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, **kwargs):
        self.max_obstacles = kwargs.get("max_obstacles", 10)
        self.include_acceleration = kwargs.get("include_acceleration", False)

        # Agent features: absolute velocity + absolute acceleration (optional)
        self._agent_size = (
            4 if self.include_acceleration else 2
        )  # [vel_x, vel_y] or [vel_x, vel_y, acc_x, acc_y]

        # Target features: position only
        self._target_size = 2  # [pos_x, pos_y]

        # Obstacle features: relative position + relative velocity + absolute acceleration (optional)
        self._obstacle_size = (
            5 if self.include_acceleration else 4
        )  # [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y] or [..., acc_x, acc_y]

        self._obstacles_total_size = self._obstacle_size * self.max_obstacles
        self._features_dim = (
            self._agent_size + self._target_size + self._obstacles_total_size
        )

        super().__init__(observation_space, self._features_dim)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        agent_data = observations[
            "agent"
        ]  # [batch_size, 7] - [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        obstacles_data = observations[
            "obstacles"
        ]  # [batch_size, max_obstacles, 7]
        target_data = observations["target"]  # [batch_size, 2] - [pos_x, pos_y]
        mask = observations[
            "mask"
        ]  # [batch_size, max_obstacles] - obstacle validity mask

        # Extract agent features (absolute velocity + optional acceleration)
        if self.include_acceleration:
            agent_features = agent_data[
                :, [3, 4, 5, 6]
            ]  # [vel_x, vel_y, acc_x, acc_y]
        else:
            agent_features = agent_data[:, [3, 4]]  # [vel_x, vel_y]

        # Target features (position only)
        target_features = target_data  # [pos_x, pos_y]

        # Extract obstacle features (relative position + relative velocity + optional acceleration)
        obstacles_features = self._encode_obstacles_relative(
            agent_data, obstacles_data, mask
        )

        # Concatenate all features
        features = torch.cat(
            [agent_features, target_features, obstacles_features], dim=1
        )
        return features

    def _encode_obstacles_relative(
        self,
        agent_data: torch.Tensor,
        obstacles_data: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = agent_data.shape[0]

        # Initialize output tensor
        output = torch.zeros(
            batch_size, self._obstacles_total_size, device=agent_data.device
        )

        # Agent position and velocity for relative calculations
        agent_pos = agent_data[:, [1, 2]]  # [pos_x, pos_y]
        agent_vel = agent_data[:, [3, 4]]  # [vel_x, vel_y]

        for i in range(self.max_obstacles):
            obstacle = obstacles_data[:, i]  # [batch_size, 7]
            obstacle_mask = mask[
                :, i
            ]  # [batch_size] - validity mask for this obstacle

            # Use mask to determine valid obstacles
            valid_mask = obstacle_mask > 0  # True where obstacle is valid

            if valid_mask.any():
                # Relative position
                obstacle_pos = obstacle[:, [1, 2]]  # [pos_x, pos_y]
                rel_pos = obstacle_pos - agent_pos

                # Relative velocity
                obstacle_vel = obstacle[:, [3, 4]]  # [vel_x, vel_y]
                rel_vel = obstacle_vel - agent_vel

                start_idx = i * self._obstacle_size

                if self.include_acceleration:
                    # [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
                    obstacle_acc = obstacle[:, [5, 6]]  # [acc_x, acc_y]
                    features = torch.cat(
                        [rel_pos, rel_vel, obstacle_acc], dim=1
                    )
                else:
                    # [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
                    features = torch.cat([rel_pos, rel_vel], dim=1)

                output[:, start_idx : start_idx + self._obstacle_size] = (
                    features * valid_mask.unsqueeze(1)
                )

        return output
