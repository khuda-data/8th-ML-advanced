"""
Utility functions for feature extractors.

This module provides common functionality for transforming observations into
relative coordinate systems and extracting features that are consistent across
all extractor implementations.
"""

import torch


def extract_agent_features(
    agent_data: torch.Tensor, include_acceleration: bool = False
) -> torch.Tensor:
    """
    Extract agent's dynamic features (velocity and optionally acceleration).

    This function extracts the agent's velocity and optionally acceleration
    from the full agent observation vector. The position and radius are
    excluded as they are not used as features in the extractor networks.

    Args:
        agent_data: Tensor of shape [batch_size, 7] containing agent observations
                   in format [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        include_acceleration: If True, includes acceleration components in output.
                            If False, only velocity components are returned.

    Returns:
        Tensor of shape [batch_size, 2] if include_acceleration=False (velocity only)
        or [batch_size, 4] if include_acceleration=True (velocity + acceleration)

    Example:
        >>> agent_data = torch.tensor([[0.5, 1.0, 2.0, 0.1, 0.2, 0.01, 0.02]])
        >>> extract_agent_features(agent_data, False)
        tensor([[0.1000, 0.2000]])
        >>> extract_agent_features(agent_data, True)
        tensor([[0.1000, 0.2000, 0.0100, 0.0200]])
    """
    if include_acceleration:
        return agent_data[:, [3, 4, 5, 6]]
    else:
        return agent_data[:, [3, 4]]


def extract_target_relative_features(
    agent_data: torch.Tensor, target_data: torch.Tensor
) -> torch.Tensor:
    """
    Extract target position relative to agent position.

    Converts absolute target position to relative position with respect to
    the agent's current position. This provides translation-invariant features
    that help the model focus on relative spatial relationships rather than
    absolute coordinates.

    Args:
        agent_data: Tensor of shape [batch_size, 7] containing agent observations
                   in format [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        target_data: Tensor of shape [batch_size, 2] containing target positions
                    in format [pos_x, pos_y]

    Returns:
        Tensor of shape [batch_size, 2] containing relative target position
        in format [rel_pos_x, rel_pos_y] where rel_pos = target_pos - agent_pos

    Example:
        >>> agent_data = torch.tensor([[0.5, 1.0, 2.0, 0.1, 0.2, 0.01, 0.02]])
        >>> target_data = torch.tensor([[5.0, 6.0]])
        >>> extract_target_relative_features(agent_data, target_data)
        tensor([[4.0000, 4.0000]])
    """
    agent_pos = agent_data[:, [1, 2]]
    target_pos = target_data
    rel_target_pos = target_pos - agent_pos
    return rel_target_pos


def extract_obstacle_relative_features(
    agent_data: torch.Tensor,
    obstacles_data: torch.Tensor,
    mask: torch.Tensor,
    include_acceleration: bool = False,
) -> torch.Tensor:
    """
    Extract obstacle features relative to agent using vectorized operations.

    This is an optimized version of extract_obstacle_relative_features that
    uses vectorized tensor operations instead of loops. It computes the same
    relative position and velocity features but with better performance for
    large batches or many obstacles. All obstacles are processed simultaneously
    using broadcasting.

    Args:
        agent_data: Tensor of shape [batch_size, 7] containing agent observations
                   in format [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        obstacles_data: Tensor of shape [batch_size, max_obstacles, 7] containing
                       obstacle observations in the same format as agent_data
        mask: Tensor of shape [batch_size, max_obstacles] with 1.0 for valid
              obstacles and 0.0 for invalid/padding obstacles
        include_acceleration: If True, includes obstacle acceleration in features.
                            If False, only relative position and velocity are used.

    Returns:
        Tensor of shape [batch_size, max_obstacles, feature_size] where:
        - feature_size = 4 if include_acceleration=False: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
        - feature_size = 5 if include_acceleration=True: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        Invalid obstacles have all features set to 0.0

    Note:
        This function produces identical results to extract_obstacle_relative_features
        but with better computational efficiency through vectorization.
    """
    agent_pos = agent_data[:, [1, 2]].unsqueeze(1)
    agent_vel = agent_data[:, [3, 4]].unsqueeze(1)

    obs_pos = obstacles_data[:, :, [1, 2]]
    obs_vel = obstacles_data[:, :, [3, 4]]

    rel_pos = obs_pos - agent_pos
    rel_vel = obs_vel - agent_vel

    if include_acceleration:
        obs_acc = obstacles_data[:, :, [5, 6]]
        features = torch.cat([rel_pos, rel_vel, obs_acc], dim=-1)
    else:
        features = torch.cat([rel_pos, rel_vel], dim=-1)

    valid_mask = (mask > 0).float().unsqueeze(-1)
    return features * valid_mask


def validate_observation_tensors(
    agent_data: torch.Tensor,
    obstacles_data: torch.Tensor,
    target_data: torch.Tensor,
    mask: torch.Tensor,
    max_obstacles: int,
) -> None:
    """
    Validate that input observation tensors have correct shapes and valid values.

    This function performs comprehensive validation of all input tensors used in
    feature extraction to ensure they meet the expected format and contain valid
    numerical values. It checks tensor shapes, data types, and value ranges to
    prevent runtime errors in downstream processing.

    Args:
        agent_data: Tensor of shape [batch_size, 7] containing agent observations
                   in format [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        obstacles_data: Tensor of shape [batch_size, max_obstacles, 7] containing
                       obstacle observations in the same format as agent_data
        target_data: Tensor of shape [batch_size, 2] containing target positions
                    in format [pos_x, pos_y]
        mask: Tensor of shape [batch_size, max_obstacles] with values in [0, 1]
              indicating obstacle validity (1.0 = valid, 0.0 = invalid/padding)
        max_obstacles: Expected maximum number of obstacles. Used to validate
                      the obstacles_data and mask tensor dimensions.

    Raises:
        ValueError: If any tensor has incorrect shape, contains NaN/Inf values,
                   or mask values are outside the valid [0, 1] range.

    Note:
        This function only validates tensor properties and does not modify
        the input tensors. It should be called before any feature extraction
        operations to ensure data integrity.
    """
    batch_size = agent_data.shape[0]

    if agent_data.shape != (batch_size, 7):
        raise ValueError(
            f"Agent data should have shape ({batch_size}, 7), got {agent_data.shape}"
        )

    if obstacles_data.shape != (batch_size, max_obstacles, 7):
        raise ValueError(
            f"Obstacles data should have shape ({batch_size}, {max_obstacles}, 7), got {obstacles_data.shape}"
        )

    if target_data.shape != (batch_size, 2):
        raise ValueError(
            f"Target data should have shape ({batch_size}, 2), got {target_data.shape}"
        )

    if mask.shape != (batch_size, max_obstacles):
        raise ValueError(
            f"Mask should have shape ({batch_size}, {max_obstacles}), got {mask.shape}"
        )

    for name, tensor in [
        ("agent", agent_data),
        ("obstacles", obstacles_data),
        ("target", target_data),
        ("mask", mask),
    ]:
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} data contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} data contains Inf values")

    if (mask < 0).any() or (mask > 1).any():
        raise ValueError("Mask values should be in range [0, 1]")
