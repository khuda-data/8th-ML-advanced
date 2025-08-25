"""Comprehensive tests for utils module.
This module tests all utility functions for feature extraction and validation.
"""

import pytest
import torch
from unittest.mock import patch

from ..utils import (
    extract_agent_features,
    extract_target_features,
    extract_obstacle_features,
    validate_observation_tensors,
)


class TestExtractAgentFeatures:
    """Test suite for extract_agent_features function."""

    def test_all_features_included(self):
        """Test extraction with all features included."""
        batch_size = 3
        # Input format: [radius, vel_x, vel_y, acc_x, acc_y]
        agent_data = torch.tensor(
            [
                [0.5, 0.1, 0.2, 0.01, 0.02],
                [0.6, 0.3, 0.4, 0.03, 0.04],
                [0.7, 0.5, 0.6, 0.05, 0.06],
            ]
        )

        result = extract_agent_features(
            agent_data, include_acceleration=True, include_radius=True
        )

        assert result.shape == (batch_size, 5)
        torch.testing.assert_close(result, agent_data)  # All features

    def test_velocity_only(self):
        """Test extraction with velocity only."""
        batch_size = 2
        agent_data = torch.tensor(
            [[0.5, 0.1, 0.2, 0.01, 0.02], [0.6, 0.3, 0.4, 0.03, 0.04]]
        )

        result = extract_agent_features(
            agent_data, include_acceleration=False, include_radius=False
        )

        assert result.shape == (batch_size, 2)
        expected = agent_data[:, 1:3]  # Only velocity
        torch.testing.assert_close(result, expected)

    def test_radius_and_velocity(self):
        """Test extraction with radius and velocity."""
        batch_size = 2
        agent_data = torch.tensor(
            [[0.5, 0.1, 0.2, 0.01, 0.02], [0.6, 0.3, 0.4, 0.03, 0.04]]
        )

        result = extract_agent_features(
            agent_data, include_acceleration=False, include_radius=True
        )

        assert result.shape == (batch_size, 3)
        expected = torch.cat([agent_data[:, 0:1], agent_data[:, 1:3]], dim=1)
        torch.testing.assert_close(result, expected)

    def test_velocity_and_acceleration(self):
        """Test extraction with velocity and acceleration."""
        batch_size = 2
        agent_data = torch.tensor(
            [[0.5, 0.1, 0.2, 0.01, 0.02], [0.6, 0.3, 0.4, 0.03, 0.04]]
        )

        result = extract_agent_features(
            agent_data, include_acceleration=True, include_radius=False
        )

        assert result.shape == (batch_size, 4)
        expected = agent_data[:, 1:5]  # velocity + acceleration
        torch.testing.assert_close(result, expected)

    def test_single_batch_item(self):
        """Test with single batch item."""
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])

        result_all = extract_agent_features(
            agent_data, include_acceleration=True, include_radius=True
        )
        result_vel_only = extract_agent_features(
            agent_data, include_acceleration=False, include_radius=False
        )

        assert result_all.shape == (1, 5)
        assert result_vel_only.shape == (1, 2)

    def test_zero_input(self):
        """Test with zero input tensor."""
        agent_data = torch.zeros(2, 5)

        result = extract_agent_features(
            agent_data, include_acceleration=True, include_radius=True
        )

        assert result.shape == (2, 5)
        assert torch.all(result == 0)


class TestExtractTargetFeatures:
    """Test suite for extract_target_features function."""

    def test_basic_target_features(self):
        """Test basic target feature extraction."""
        batch_size = 2
        # target_data: [rel_x, rel_y] - already relative position
        target_data = torch.tensor(
            [[0.4, 0.6], [-0.2, -0.3]]  # Pre-computed relative position
        )

        result = extract_target_features(target_data)

        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(result, target_data)  # Should return as-is

    def test_zero_position(self):
        """Test when target position is zero."""
        target_data = torch.tensor([[0.0, 0.0]])  # Target at agent position

        result = extract_target_features(target_data)

        expected = torch.zeros(1, 2)
        torch.testing.assert_close(result, expected)

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        target_data = torch.tensor([[-0.5, -0.8]])  # Target behind agent

        result = extract_target_features(target_data)

        expected = torch.tensor([[-0.5, -0.8]])
        torch.testing.assert_close(result, expected)

    def test_multiple_batch_items(self):
        """Test with multiple batch items."""
        batch_size = 4
        target_data = torch.tensor(
            [[1.0, 1.5], [-0.5, 0.8], [0.0, 0.0], [2.5, -1.2]]
        )

        result = extract_target_features(target_data)

        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(result, target_data)


class TestExtractObstacleFeatures:
    """Test suite for extract_obstacle_features function."""

    def test_without_acceleration_without_radius(self):
        """Test obstacle extraction with velocity only."""
        batch_size = 2
        max_obstacles = 3

        # obstacles_data format: [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, -0.5, -0.5, 1.0, 2.0],  # Valid obstacle
                    [0.3, 2.0, 2.0, -0.3, -0.4, 3.0, 4.0],  # Valid obstacle
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Invalid (masked)
                ],
                [
                    [0.4, -1.0, 0.5, 0.2, -0.1, -1.0, 0.5],  # Valid obstacle
                    [0.2, 0.8, -0.3, 0.1, 0.3, 2.0, -1.0],  # Valid obstacle
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Invalid (masked)
                ],
            ]
        )

        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

        result = extract_obstacle_features(
            obstacles_data,
            mask,
            include_acceleration=False,
            include_radius=False,
        )

        assert result.shape == (batch_size, max_obstacles, 4)  # pos + vel only

        # Check features: [rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
        expected_features = obstacles_data[
            :, :, 1:5
        ]  # Skip radius, include pos+vel
        torch.testing.assert_close(
            result[:, :2, :], expected_features[:, :2, :]
        )

        # Check masked obstacle should be zeros
        torch.testing.assert_close(result[:, 2, :], torch.zeros(batch_size, 4))

    def test_with_acceleration_with_radius(self):
        """Test obstacle extraction with all features."""
        batch_size = 1
        max_obstacles = 2

        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, -0.5, -0.5, 2.0, 3.0],  # Valid obstacle
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Invalid (masked)
                ]
            ]
        )

        mask = torch.tensor([[1.0, 0.0]])

        result = extract_obstacle_features(
            obstacles_data, mask, include_acceleration=True, include_radius=True
        )

        assert result.shape == (batch_size, max_obstacles, 7)  # All features

        # Check valid obstacle keeps all features
        torch.testing.assert_close(result[0, 0, :], obstacles_data[0, 0, :])

        # Check masked obstacle should be zeros
        torch.testing.assert_close(result[0, 1, :], torch.zeros(7))

    def test_radius_only(self):
        """Test obstacle extraction with radius and position only."""
        batch_size = 1
        max_obstacles = 2

        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, -0.5, -0.5, 2.0, 3.0],  # Valid obstacle
                    [0.4, 2.0, 2.0, -0.3, -0.4, 1.0, 2.0],  # Valid obstacle
                ]
            ]
        )

        mask = torch.tensor([[1.0, 1.0]])

        result = extract_obstacle_features(
            obstacles_data,
            mask,
            include_acceleration=False,
            include_radius=True,
        )

        assert result.shape == (
            batch_size,
            max_obstacles,
            5,
        )  # radius + pos + vel

        # Check features: [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y]
        expected_features = obstacles_data[:, :, :5]
        torch.testing.assert_close(result, expected_features)

    def test_empty_mask(self):
        """Test with all obstacles masked out."""
        obstacles_data = torch.randn(1, 3, 7)
        mask = torch.zeros(1, 3)

        result = extract_obstacle_features(
            obstacles_data,
            mask,
            include_acceleration=False,
            include_radius=False,
        )

        assert result.shape == (1, 3, 4)
        torch.testing.assert_close(result, torch.zeros(1, 3, 4))

    def test_partial_mask(self):
        """Test with partial masking."""
        batch_size = 2
        max_obstacles = 4
        obstacles_data = torch.randn(batch_size, max_obstacles, 7)

        # Different mask patterns for each batch
        mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],  # First batch: 2 valid
                [1.0, 0.0, 1.0, 0.0],  # Second batch: 2 valid (non-contiguous)
            ]
        )

        result = extract_obstacle_features(
            obstacles_data, mask, include_acceleration=True, include_radius=True
        )

        assert result.shape == (batch_size, max_obstacles, 7)

        # Check that masked positions are zero
        torch.testing.assert_close(
            result[0, 2:, :], torch.zeros(2, 7)
        )  # Last 2 in first batch
        torch.testing.assert_close(
            result[1, 1, :], torch.zeros(7)
        )  # 2nd in second batch
        torch.testing.assert_close(
            result[1, 3, :], torch.zeros(7)
        )  # Last in second batch


if __name__ == "__main__":
    pytest.main([__file__])
