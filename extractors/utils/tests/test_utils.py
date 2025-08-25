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
    validate_observation_tensors,
)


class TestExtractAgentFeatures:
    """Test suite for extract_agent_features function."""

    def test_extract_without_acceleration(self):
        """Test extraction without acceleration features."""
        batch_size = 3
        # New format: [radius, vel_x, vel_y, acc_x, acc_y]
        agent_data = torch.tensor(
            [
                [0.5, 0.1, 0.2, 0.01, 0.02],
                [0.6, 0.3, 0.4, 0.03, 0.04],
                [0.7, 0.5, 0.6, 0.05, 0.06],
            ]
        )

        result = extract_agent_features(agent_data, include_acceleration=False)

        assert result.shape == (batch_size, 3)
        torch.testing.assert_close(
            result, agent_data[:, :3]
        )  # radius, vel_x, vel_y

    def test_extract_with_acceleration(self):
        """Test extraction with acceleration features."""
        batch_size = 3
        # New format: [radius, vel_x, vel_y, acc_x, acc_y]
        agent_data = torch.tensor(
            [
                [0.5, 0.1, 0.2, 0.01, 0.02],
                [0.6, 0.3, 0.4, 0.03, 0.04],
                [0.7, 0.5, 0.6, 0.05, 0.06],
            ]
        )

        result = extract_agent_features(agent_data, include_acceleration=True)

        assert result.shape == (batch_size, 5)
        torch.testing.assert_close(result, agent_data)  # All features

    def test_single_batch_item(self):
        """Test with single batch item."""
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])

        result_without_acc = extract_agent_features(
            agent_data, include_acceleration=False
        )
        result_with_acc = extract_agent_features(
            agent_data, include_acceleration=True
        )

        assert result_without_acc.shape == (1, 3)
        assert result_with_acc.shape == (1, 5)

    def test_zero_input(self):
        """Test with zero input tensor."""
        agent_data = torch.zeros(2, 5)

        result = extract_agent_features(agent_data, include_acceleration=False)

        assert result.shape == (2, 3)
        assert torch.all(result == 0)


class TestExtractTargetRelativeFeatures:
    """Test suite for extract_target_relative_features function."""

    def test_basic_relative_position(self):
        """Test basic relative position calculation."""
        batch_size = 2
        # agent_data not used in new structure, but kept for compatibility
        agent_data = torch.tensor(
            [
                [0.5, 0.1, 0.2, 0.01, 0.02],
                [0.6, 0.3, 0.4, 0.03, 0.04],
            ]
        )
        # target_data is already relative position
        target_data = torch.tensor(
            [[0.4, 0.6], [-0.2, -0.3]]  # Pre-computed relative position
        )

        result = extract_target_relative_features(agent_data, target_data)

        assert result.shape == (batch_size, 2)
        torch.testing.assert_close(result, target_data)  # Should return as-is

    def test_zero_relative_position(self):
        """Test when target relative position is zero."""
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])
        target_data = torch.tensor([[0.0, 0.0]])  # Target at agent position

        result = extract_target_relative_features(agent_data, target_data)

        expected = torch.zeros(1, 2)
        torch.testing.assert_close(result, expected)

    def test_negative_coordinates(self):
        """Test with negative relative coordinates."""
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])
        target_data = torch.tensor([[-0.5, -0.8]])  # Target behind agent

        result = extract_target_relative_features(agent_data, target_data)

        expected = torch.tensor([[-0.5, -0.8]])
        torch.testing.assert_close(result, expected)


class TestExtractObstacleRelativeFeatures:
    """Test suite for extract_obstacle_relative_features function."""

    def test_without_acceleration(self):
        """Test obstacle extraction without acceleration."""
        batch_size = 2
        max_obstacles = 3

        # agent_data not used in new structure, but kept for compatibility
        agent_data = torch.tensor(
            [
                [0.5, 0.1, 0.2, 0.01, 0.02],
                [0.6, 0.3, 0.4, 0.03, 0.04],
            ]
        )

        # obstacles_data already contains relative features: [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
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

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=False
        )

        assert result.shape == (
            batch_size,
            max_obstacles,
            5,
        )  # First 5 features only

        # Check valid obstacles keep their features (first 5 only)
        torch.testing.assert_close(result[0, 0, :], obstacles_data[0, 0, :5])
        torch.testing.assert_close(result[0, 1, :], obstacles_data[0, 1, :5])

        # Check masked obstacle should be zeros
        torch.testing.assert_close(result[0, 2, :], torch.zeros(5))

    def test_with_acceleration(self):
        """Test obstacle extraction including acceleration."""
        batch_size = 1
        max_obstacles = 2

        # agent_data not used in new structure, but kept for compatibility
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])

        # obstacles_data already contains all features: [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y]
        obstacles_data = torch.tensor(
            [
                [
                    [0.3, 1.0, 1.0, -0.5, -0.5, 2.0, 3.0],  # Valid obstacle
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Invalid (masked)
                ]
            ]
        )

        mask = torch.tensor([[1.0, 0.0]])

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=True
        )

        assert result.shape == (batch_size, max_obstacles, 7)  # All 7 features

        # Check valid obstacle keeps all features
        torch.testing.assert_close(result[0, 0, :], obstacles_data[0, 0, :])

        # Check masked obstacle should be zeros
        torch.testing.assert_close(result[0, 1, :], torch.zeros(7))

    def test_empty_mask(self):
        """Test with all obstacles masked out."""
        agent_data = torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]])
        obstacles_data = torch.randn(1, 3, 7)
        mask = torch.zeros(1, 3)

        result = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, include_acceleration=False
        )

        assert result.shape == (1, 3, 5)
        torch.testing.assert_close(result, torch.zeros(1, 3, 5))


class TestValidateObservationTensors:
    """Test suite for validate_observation_tensors function."""

    def test_valid_tensors_without_acceleration(self):
        """Test validation with valid tensors (without acceleration)."""
        batch_size = 2
        max_obstacles = 3

        agent_data = torch.randn(
            batch_size, 3
        )  # Without acceleration: 3 features
        obstacles_data = torch.randn(
            batch_size, max_obstacles, 5
        )  # Without acceleration: 5 features
        target_data = torch.randn(batch_size, 2)
        mask = torch.rand(batch_size, max_obstacles)

        # Should not raise any exception
        validate_observation_tensors(
            agent_data,
            obstacles_data,
            target_data,
            mask,
            max_obstacles,
            include_acceleration=False,
        )

    def test_valid_tensors_with_acceleration(self):
        """Test validation with valid tensors (with acceleration)."""
        batch_size = 2
        max_obstacles = 3

        agent_data = torch.randn(batch_size, 5)  # With acceleration: 5 features
        obstacles_data = torch.randn(
            batch_size, max_obstacles, 7
        )  # With acceleration: 7 features
        target_data = torch.randn(batch_size, 2)
        mask = torch.rand(batch_size, max_obstacles)

        # Should not raise any exception
        validate_observation_tensors(
            agent_data,
            obstacles_data,
            target_data,
            mask,
            max_obstacles,
            include_acceleration=True,
        )

    def test_invalid_agent_shape_without_acceleration(self):
        """Test validation with invalid agent shape (without acceleration)."""
        agent_data = torch.randn(2, 5)  # Wrong feature size (should be 3)
        obstacles_data = torch.randn(2, 3, 5)
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 3)

        with pytest.raises(ValueError, match="Agent data should have shape"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=False,
            )

    def test_invalid_agent_shape_with_acceleration(self):
        """Test validation with invalid agent shape (with acceleration)."""
        agent_data = torch.randn(2, 3)  # Wrong feature size (should be 5)
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 3)

        with pytest.raises(ValueError, match="Agent data should have shape"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )

    def test_invalid_obstacles_shape(self):
        """Test validation with invalid obstacles shape."""
        agent_data = torch.randn(2, 5)
        obstacles_data = torch.randn(2, 5, 7)  # Wrong max_obstacles
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 3)

        with pytest.raises(
            ValueError, match="Obstacles data should have shape"
        ):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )

    def test_invalid_target_shape(self):
        """Test validation with invalid target shape."""
        agent_data = torch.randn(2, 5)
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 3)  # Wrong feature size
        mask = torch.ones(2, 3)

        with pytest.raises(ValueError, match="Target data should have shape"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )

    def test_invalid_mask_shape(self):
        """Test validation with invalid mask shape."""
        agent_data = torch.randn(2, 5)
        obstacles_data = torch.randn(2, 3, 7)
        target_data = torch.randn(2, 2)
        mask = torch.ones(2, 5)  # Wrong max_obstacles

        with pytest.raises(ValueError, match="Mask should have shape"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )

    def test_nan_values(self):
        """Test validation with NaN values."""
        agent_data = torch.tensor([[float("nan"), 0, 0]], dtype=torch.float32)
        obstacles_data = torch.randn(1, 3, 5)
        target_data = torch.randn(1, 2)
        mask = torch.ones(1, 3)

        with pytest.raises(ValueError, match="agent data contains NaN values"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=False,
            )

    def test_inf_values(self):
        """Test validation with Inf values."""
        agent_data = torch.randn(1, 5)
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
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )

    def test_invalid_mask_range(self):
        """Test validation with mask values outside [0, 1] range."""
        agent_data = torch.randn(1, 5)
        obstacles_data = torch.randn(1, 3, 7)
        target_data = torch.randn(1, 2)
        mask = torch.tensor(
            [[1.0, 0.5, -0.1]], dtype=torch.float32
        )  # Negative value

        with pytest.raises(ValueError, match="Mask values should be in range"):
            validate_observation_tensors(
                agent_data,
                obstacles_data,
                target_data,
                mask,
                3,
                include_acceleration=True,
            )


if __name__ == "__main__":
    pytest.main([__file__])
