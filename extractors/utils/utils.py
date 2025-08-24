"""
Utility functions for feature extractors.

This module provides common functionality for transforming observations into
relative coordinate systems and extracting features that are consistent across
all extractor implementations.
"""

import torch
from typing import Tuple


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
    Extract obstacle features relative to agent using iterative approach.

    This function processes obstacles one by one in a loop, computing relative
    position and velocity for each obstacle with respect to the agent. Invalid
    obstacles (marked by mask=0) are set to zero. This is the non-vectorized
    version that may be slower but easier to understand and debug.

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
        For better performance with large numbers of obstacles, consider using
        extract_obstacle_relative_features_vectorized instead.
    """
    batch_size, max_obstacles = obstacles_data.shape[:2]
    feature_size = 6 if include_acceleration else 4

    output = torch.zeros(
        batch_size, max_obstacles, feature_size, device=agent_data.device
    )

    agent_pos = agent_data[:, [1, 2]]
    agent_vel = agent_data[:, [3, 4]]

    for i in range(max_obstacles):
        obstacle = obstacles_data[:, i]
        obstacle_mask = mask[:, i]

        valid_mask = obstacle_mask > 0

        if valid_mask.any():
            obstacle_pos = obstacle[:, [1, 2]]
            rel_pos = obstacle_pos - agent_pos

            obstacle_vel = obstacle[:, [3, 4]]
            rel_vel = obstacle_vel - agent_vel

            if include_acceleration:
                obstacle_acc = obstacle[:, [5, 6]]
                features = torch.cat([rel_pos, rel_vel, obstacle_acc], dim=1)
            else:
                features = torch.cat([rel_pos, rel_vel], dim=1)

            output[:, i] = features * valid_mask.unsqueeze(1)

    return output


def extract_obstacle_relative_features_vectorized(
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


def flatten_obstacle_features(
    obstacle_features: torch.Tensor, max_obstacles: int, feature_size: int
) -> torch.Tensor:
    """
    Flatten obstacle features into a single vector for padding-based processing.

    This function converts the 3D obstacle feature tensor into a 2D tensor by
    flattening the obstacle and feature dimensions. This is commonly used in
    padding-based extractors where all obstacle features are concatenated into
    a single vector for further processing.

    Args:
        obstacle_features: Tensor of shape [batch_size, max_obstacles, feature_size]
                          containing the extracted obstacle features
        max_obstacles: Maximum number of obstacles that can be present. This should
                      match the second dimension of obstacle_features
        feature_size: Number of features per obstacle. This should match the third
                     dimension of obstacle_features

    Returns:
        Tensor of shape [batch_size, max_obstacles * feature_size] containing
        the flattened obstacle features where all obstacle features are concatenated
        sequentially for each batch item.

    Example:
        If obstacle_features has shape [2, 3, 4] (2 batches, 3 obstacles, 4 features each),
        the output will have shape [2, 12] where each row contains all obstacle
        features concatenated: [obs1_feat1, obs1_feat2, ..., obs3_feat4]
    """
    batch_size = obstacle_features.shape[0]
    return obstacle_features.reshape(batch_size, max_obstacles * feature_size)


def get_feature_dimensions(
    max_obstacles: int, include_acceleration: bool = False
) -> Tuple[int, int, int, int]:
    """
    Calculate standard feature dimensions used across all extractor types.

    This function provides consistent feature dimension calculations for different
    extractor architectures. It accounts for whether acceleration features are
    included and returns all relevant dimension sizes needed for feature extraction
    and neural network layer sizing.

    Args:
        max_obstacles: Maximum number of obstacles that can be present in the
                      environment. This determines the total size when obstacles
                      are flattened for padding-based processing.
        include_acceleration: If True, includes acceleration features in the
                            dimension calculations. If False, only position
                            and velocity features are counted.

    Returns:
        A tuple containing four dimension sizes:
        - agent_size: Number of features extracted per agent (2 or 4)
        - target_size: Number of features extracted per target (always 2)
        - obstacle_size: Number of features extracted per obstacle (4 or 6)
        - obstacles_total_size: Total flattened size for all obstacles (obstacle_size * max_obstacles)

    Feature Breakdown:
        - Agent features: [vel_x, vel_y] + optional [acc_x, acc_y] (absolute coordinates)
        - Target features: [rel_pos_x, rel_pos_y] (relative to agent)
        - Obstacle features: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y] + optional [acc_x, acc_y]
    """
    agent_size = 4 if include_acceleration else 2
    target_size = 2
    obstacle_size = 6 if include_acceleration else 4
    obstacles_total_size = obstacle_size * max_obstacles

    return agent_size, target_size, obstacle_size, obstacles_total_size


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


def compute_sequence_lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the number of valid obstacles per batch for LSTM sequence processing.

    This function calculates the actual sequence length for each batch item
    by counting the number of valid obstacles (non-zero mask values). This
    is essential for LSTM processing where sequences have variable lengths
    and need to be packed efficiently to avoid processing padding tokens.

    Args:
        mask: Tensor of shape [batch_size, max_obstacles] containing obstacle
              validity indicators. Values should be 1.0 for valid obstacles
              and 0.0 for invalid/padding obstacles. Can be floating point
              or integer tensor.

    Returns:
        Tensor of shape [batch_size] containing the number of valid obstacles
        for each batch item. Minimum value is clamped to 1 to prevent empty
        sequence issues in LSTM processing.

    Note:
        For floating point masks, values > 0.5 are considered valid.
        For integer masks, any non-zero value is considered valid.
        The minimum length of 1 ensures LSTM doesn't receive empty sequences.
    """
    if mask.dtype.is_floating_point:
        lengths = (mask > 0.5).sum(dim=1)
    else:
        lengths = mask.long().sum(dim=1)

    return lengths.clamp(min=1)
