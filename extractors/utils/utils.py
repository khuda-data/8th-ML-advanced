"""
Utility functions for feature extractors.

This module provides common functionality for transforming observations into
relative coordinate systems and extracting features that are consistent across
all extractor implementations.
"""

import torch


def extract_agent_features(
    agent_data: torch.Tensor,
    include_acceleration: bool = False,
    include_radius: bool = True,
) -> torch.Tensor:
    """
    Extract agent's features from pre-processed observation.

    The new observation structure provides agent features in the format:
    [radius, vel_x, vel_y, acc_x, acc_y] (all pre-scaled)

    Args:
        agent_data: Tensor of shape [batch_size, 5] containing agent observations
                   in format [radius, vel_x, vel_y, acc_x, acc_y] (pre-scaled)
        include_acceleration: If True, includes acceleration features
        include_radius: If True, includes radius feature

    Returns:
        Tensor with selected features based on parameters

    Example:
        >>> agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])
        >>> extract_agent_features(agent_data, include_acceleration=True, include_radius=True)
        tensor([[0.5000, 0.1000, 0.2000, 0.0100, 0.0200]])
        >>> extract_agent_features(agent_data, include_acceleration=False, include_radius=False)
        tensor([[0.1000, 0.2000]])
    """
    features = []

    if include_radius:
        features.append(agent_data[:, 0:1])  # radius

    features.append(agent_data[:, 1:3])  # vel_x, vel_y

    if include_acceleration:
        features.append(agent_data[:, 3:5])  # acc_x, acc_y

    return torch.cat(features, dim=1)


def extract_target_features(target_data: torch.Tensor) -> torch.Tensor:
    """
    Extract target position features from pre-processed observation.

    The new observation structure already provides target position relative to agent,
    scaled by recognition radius: [rel_pos_x, rel_pos_y]

    Args:
        target_data: Tensor of shape [batch_size, 2] containing target relative position
                    in format [rel_pos_x, rel_pos_y] (pre-scaled)

    Returns:
        Tensor of shape [batch_size, 2] containing pre-processed relative target position

    Example:
        >>> target_data = torch.tensor([[0.4, 0.6]])  # Pre-scaled relative position
        >>> extract_target_features(target_data)
        tensor([[0.4000, 0.6000]])
    """

    return target_data


def extract_obstacle_features(
    obstacles_data: torch.Tensor,
    mask: torch.Tensor,
    include_acceleration: bool = False,
    include_radius: bool = True,
) -> torch.Tensor:
    """
    Extract obstacle features from pre-processed observation data.

    The new observation structure provides obstacles with:
    [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y] (all pre-scaled)

    Args:
        obstacles_data: Tensor of shape [batch_size, max_obstacles, 7] containing
                       obstacle observations in format:
                       [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        mask: Tensor of shape [batch_size, max_obstacles] with 1.0 for valid
              obstacles and 0.0 for invalid/padding obstacles
        include_acceleration: If True, includes acceleration features
        include_radius: If True, includes radius feature

    Returns:
        Tensor with selected features based on parameters
        Invalid obstacles have all features set to 0.0

    Example:
        >>> obstacles_data = torch.tensor([[[1.0, 0.1, 0.2, 0.01, 0.02, 0.001, 0.002]]])
        >>> mask = torch.tensor([[1.0]])
        >>> extract_obstacle_features(obstacles_data, mask, True, True)
        tensor([[[1.0000, 0.1000, 0.2000, 0.0100, 0.0200, 0.0010, 0.0020]]])
        >>> extract_obstacle_features(obstacles_data, mask, False, False)
        tensor([[[0.1000, 0.2000, 0.0100, 0.0200]]])
    """
    valid_mask = (mask > 0).float().unsqueeze(-1)

    features = []

    if include_radius:
        features.append(obstacles_data[:, :, 0:1])  # radius

    features.append(
        obstacles_data[:, :, 1:5]
    )  # rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y

    if include_acceleration:
        features.append(obstacles_data[:, :, 5:7])  # acc_x, acc_y

    selected_features = torch.cat(features, dim=-1)

    return selected_features * valid_mask


def validate_observation_tensors(
    agent_data: torch.Tensor,
    obstacles_data: torch.Tensor,
    target_data: torch.Tensor,
    mask: torch.Tensor,
    max_obstacles: int,
    include_acceleration: bool = False,
    include_radius: bool = True,
) -> None:
    """
    Validate that input observation tensors have correct shapes and valid values.

    Updated for the new observation structure with pre-processed features.

    Args:
        agent_data: Tensor containing agent observations (pre-scaled)
        obstacles_data: Tensor containing obstacle observations (pre-scaled)
        target_data: Tensor of shape [batch_size, 2] containing target relative position
                    in format [rel_pos_x, rel_pos_y] (pre-scaled)
        mask: Tensor of shape [batch_size, max_obstacles] with values in [0, 1]
              indicating obstacle validity (1.0 = valid, 0.0 = invalid/padding)
        max_obstacles: Expected maximum number of obstacles
        include_acceleration: If True, includes acceleration features
        include_radius: If True, includes radius features

    Raises:
        ValueError: If any tensor has incorrect shape, contains NaN/Inf values,
                   or mask values are outside the valid [0, 1] range.
    """
    batch_size = agent_data.shape[0]

    # Calculate expected feature sizes based on flags
    agent_features = 2  # Base: vel_x, vel_y
    if include_radius:
        agent_features += 1  # Add radius
    if include_acceleration:
        agent_features += 2  # Add acc_x, acc_y

    obstacle_features = 4  # Base: rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y
    if include_radius:
        obstacle_features += 1  # Add radius
    if include_acceleration:
        obstacle_features += 2  # Add acc_x, acc_y

    if agent_data.shape != (batch_size, agent_features):
        raise ValueError(
            f"Agent data should have shape ({batch_size}, {agent_features}), got {agent_data.shape}"
        )

    if obstacles_data.shape != (
        batch_size,
        max_obstacles,
        obstacle_features,
    ):
        raise ValueError(
            f"Obstacles data should have shape ({batch_size}, {max_obstacles}, {obstacle_features}), got {obstacles_data.shape}"
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
