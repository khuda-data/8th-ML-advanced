"""
Comprehensive tests for utils.py functions.

This module tests all utility functions used across extractors including
feature extraction, coordinate transformations, validation, and helper functions.
"""

import pytest
import torch
import numpy as np
from typing import Tuple

from ..utils import (
    extract_agent_features,
    extract_target_relative_features,
    extract_obstacle_relative_features,
    flatten_obstacle_features,
    get_feature_dimensions,
    validate_observation_tensors,
    compute_sequence_lengths_from_mask,
)


class TestExtractAgentFeatures:
    """Test suite for extract_agent_features function."""

    def test_extract_velocity_only(self):
        """Test extraction of velocity features only."""
        batch_size = 3
        agent_data = torch.randn(batch_size, 7)

        result = extract_agent_features(agent_data, include_acceleration=False)

        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(result, agent_data[:, 3:5])

    def test_extract_velocity_and_acceleration(self):
        """Test extraction of velocity and acceleration features."""
        batch_size = 3
        agent_data = torch.randn(batch_size, 7)

        result = extract_agent_features(agent_data, include_acceleration=True)

        assert result.shape == (batch_size, 4)
        torch.testing.assert_close(result, agent_data[:, 3:7])

    def test_single_batch_item(self):
        """Test with single batch item."""
        agent_data = torch.randn(1, 7)

        result_vel_only = extract_agent_features(
            agent_data, include_acceleration=False
        )
        result_vel_acc = extract_agent_features(
            agent_data, include_acceleration=True
        )

        assert result_vel_only.shape == (1, 2)
        assert result_vel_acc.shape == (1, 4)

    def test_zero_input(self):
        """Test with zero input tensor."""
        agent_data = torch.zeros(2, 7)

        result = extract_agent_features(agent_data, include_acceleration=False)

        assert result.shape == (2, 2)
        assert torch.all(result == 0)


class TestExtractTargetRelativeFeatures:
    """Test suite for extract_target_relative_features function."""

    def test_basic_relative_position(self):
        """Test basic relative position calculation."""
        batch_size = 2
        agent_data = torch.tensor(
            [
                [0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        target_data = torch.tensor(
            [[5.0, 6.0], [1.0, 2.0]], dtype=torch.float32
        )

        result = extract_target_relative_features(agent_data, target_data)

        expected = torch.tensor([[4.0, 4.0], [-2.0, -2.0]], dtype=torch.float32)
        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(result, expected)

    def test_zero_relative_position(self):
        """Test when agent and target are at same position."""
        agent_data = torch.tensor(
            [[0.5, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        target_data = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        result = extract_target_relative_features(agent_data, target_data)

        expected = torch.zeros(1, 2)
        torch.testing.assert_close(result, expected)

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        agent_data = torch.tensor(
            [[0.5, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        target_data = torch.tensor([[-3.0, 1.0]], dtype=torch.float32)

        result = extract_target_relative_features(agent_data, target_data)

        expected = torch.tensor([[-2.0, 3.0]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)


class TestExtractObstacleRelativeFeatures:
    """Test suite for extract_obstacle_relative_features function."""

    def test_basic_obstacle_extraction(self):
        """Test basic obstacle feature extraction."""
        batch_size = 2
        max_obstacles = 3

        agent_data = torch.tensor(
            [
                [0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.5, 2.0, 2.0, -1.0, -1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0],
                    [0.3, 2.0, 0.0, -0.5, 1.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.3, 3.0, 1.0, 0.0, 2.0, 0.0, 0.0],
                    [0.3, 1.0, 3.0, 1.0, -2.0, 0.0, 0.0],
                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=torch.float32,
        )

        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32
        )

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=False
        )

        assert result.shape == (batch_size, max_obstacles, 4)

        # Check first batch item, first obstacle
        expected_rel_pos_0_0 = torch.tensor(
            [1.0, 1.0], dtype=torch.float32
        )  # obstacle_pos - agent_pos
        expected_rel_vel_0_0 = torch.tensor(
            [-0.5, -0.5], dtype=torch.float32
        )  # obstacle_vel - agent_vel
        torch.testing.assert_close(result[0, 0, :2], expected_rel_pos_0_0)
        torch.testing.assert_close(result[0, 0, 2:4], expected_rel_vel_0_0)

        # Check masked obstacle should be zeros
        torch.testing.assert_close(result[0, 2, :], torch.zeros(4))

    def test_with_acceleration(self):
        """Test obstacle extraction including acceleration."""
        batch_size = 1
        max_obstacles = 2

        agent_data = torch.tensor(
            [[0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], dtype=torch.float32
        )
        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, 0.0, 0.0, 2.0, 3.0],
                    [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=True
        )

        assert result.shape == (batch_size, max_obstacles, 6)

        # Check acceleration is included
        expected_acc = torch.tensor([2.0, 3.0], dtype=torch.float32)
        torch.testing.assert_close(result[0, 0, 4:6], expected_acc)

    def test_empty_mask(self):
        """Test with all obstacles masked out."""
        agent_data = torch.randn(1, 7)
        obstacles_data = torch.randn(1, 3, 7)
        mask = torch.zeros(1, 3)

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=False
        )

        assert result.shape == (1, 3, 4)
        torch.testing.assert_close(result, torch.zeros(1, 3, 4))


class TestFlattenObstacleFeatures:
    """Test suite for flatten_obstacle_features function."""

    def test_basic_flattening(self):
        """Test basic flattening functionality."""
        batch_size = 2
        max_obstacles = 3
        feature_size = 4

        obstacle_features = torch.randn(batch_size, max_obstacles, feature_size)

        result = flatten_obstacle_features(
            obstacle_features, max_obstacles, feature_size
        )

        assert result.shape == (batch_size, max_obstacles * feature_size)

        # Check that reshape is correct
        expected = obstacle_features.reshape(batch_size, -1)
        torch.testing.assert_close(result, expected)

    def test_single_obstacle(self):
        """Test with single obstacle."""
        obstacle_features = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32
        )

        result = flatten_obstacle_features(obstacle_features, 1, 4)

        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_zero_features(self):
        """Test with zero features."""
        obstacle_features = torch.zeros(2, 3, 5)

        result = flatten_obstacle_features(obstacle_features, 3, 5)

        assert result.shape == (2, 15)
        torch.testing.assert_close(result, torch.zeros(2, 15))


class TestGetFeatureDimensions:
    """Test suite for get_feature_dimensions function."""

    def test_dimensions_without_acceleration(self):
        """Test feature dimensions without acceleration."""
        max_obstacles = 5

        agent_size, target_size, obstacle_size, obstacles_total_size = (
            get_feature_dimensions(max_obstacles, include_acceleration=False)
        )

        assert agent_size == 2  # [vel_x, vel_y]
        assert target_size == 2  # [rel_pos_x, rel_pos_y]
        assert (
            obstacle_size == 4
        )  # [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
        assert obstacles_total_size == 20  # 4 * 5

    def test_dimensions_with_acceleration(self):
        """Test feature dimensions with acceleration."""
        max_obstacles = 3

        agent_size, target_size, obstacle_size, obstacles_total_size = (
            get_feature_dimensions(max_obstacles, include_acceleration=True)
        )

        assert agent_size == 4  # [vel_x, vel_y, acc_x, acc_y]
        assert target_size == 2  # [rel_pos_x, rel_pos_y]
        assert (
            obstacle_size == 6
        )  # [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        assert obstacles_total_size == 18  # 6 * 3

    def test_zero_obstacles(self):
        """Test with zero obstacles."""
        agent_size, target_size, obstacle_size, obstacles_total_size = (
            get_feature_dimensions(0, include_acceleration=False)
        )

        assert obstacles_total_size == 0


class TestValidateObservationTensors:
    """Test suite for validate_observation_tensors function."""

    def test_valid_tensors(self):
        """Test validation with valid tensors."""
        batch_size = 2
        max_obstacles = 3

        agent_data = torch.randn(batch_size, 7)
        obstacles_data = torch.randn(batch_size, max_obstacles, 7)
        target_data = torch.randn(batch_size, 2)
        mask = torch.rand(batch_size, max_obstacles)

        # Should not raise any exception
        validate_observation_tensors(
            agent_data, obstacles_data, target_data, mask, max_obstacles
        )

    def test_invalid_agent_shape(self):
        """Test validation with invalid agent shape."""
        agent_data = torch.randn(2, 5)  # Wrong feature size
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 3)

        with pytest.raises(ValueError, match="Agent data should have shape"):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_invalid_obstacles_shape(self):
        """Test validation with invalid obstacles shape."""
        agent_data = torch.randn(2, 7)
        obstacles_data = torch.randn(2, 5, 7)  # Wrong max_obstacles
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 3)

        with pytest.raises(
            ValueError, match="Obstacles data should have shape"
        ):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_invalid_target_shape(self):
        """Test validation with invalid target shape."""
        agent_data = torch.randn(2, 7)
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 3)  # Wrong feature size
        mask = torch.ones(2, 3)

        with pytest.raises(ValueError, match="Target data should have shape"):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_invalid_mask_shape(self):
        """Test validation with invalid mask shape."""
        agent_data = torch.randn(2, 7)
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 5)  # Wrong max_obstacles

        with pytest.raises(ValueError, match="Mask should have shape"):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_nan_values(self):
        """Test validation with NaN values."""
        agent_data = torch.tensor(
            [[float("nan"), 0, 0, 0, 0, 0, 0]], dtype=torch.float32
        )
        obstacles_data = torch.randn(1, 3, 7)
        target_data = torch.randn(1, 2)
        mask = torch.ones(1, 3)

        with pytest.raises(ValueError, match="agent data contains NaN values"):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_inf_values(self):
        """Test validation with Inf values."""
        agent_data = torch.randn(1, 7)
        obstacles_data = torch.tensor(
            [
                [
                    [float("inf"), 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ],
            dtype=torch.float32,
        )
        target_data = torch.randn(1, 2)
        mask = torch.ones(1, 3)

        with pytest.raises(
            ValueError, match="obstacles data contains Inf values"
        ):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )

    def test_invalid_mask_range(self):
        """Test validation with mask values outside [0, 1] range."""
        agent_data = torch.randn(1, 7)
        obstacles_data = torch.randn(1, 3, 7)
        target_data = torch.randn(1, 2)
        mask = torch.tensor(
            [[1.0, 0.5, -0.1]], dtype=torch.float32
        )  # Negative value

        with pytest.raises(ValueError, match="Mask values should be in range"):
            validate_observation_tensors(
                agent_data, obstacles_data, target_data, mask, 3
            )


class TestComputeSequenceLengthsFromMask:
    """Test suite for compute_sequence_lengths_from_mask function."""

    def test_floating_point_mask(self):
        """Test with floating point mask."""
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
        )

        result = compute_sequence_lengths_from_mask(mask)

        expected = torch.tensor([2, 1], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_integer_mask(self):
        """Test with integer mask."""
        mask = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.long)

        result = compute_sequence_lengths_from_mask(mask)

        expected = torch.tensor([3, 2], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_all_zeros_mask(self):
        """Test with all zeros mask (should return minimum length of 1)."""
        mask = torch.zeros(2, 3)

        result = compute_sequence_lengths_from_mask(mask)

        expected = torch.tensor([1, 1], dtype=torch.long)
        torch.testing.assert_close(result, expected)

    def test_partial_values_mask(self):
        """Test with partial values in floating point mask."""
        mask = torch.tensor(
            [[0.7, 0.3, 0.9], [0.6, 0.4, 0.1]], dtype=torch.float32
        )

        result = compute_sequence_lengths_from_mask(mask)

        # Values > 0.5 are counted as valid
        expected = torch.tensor([2, 1], dtype=torch.long)
        torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
