"""
Comprehensive tests for PaddingExtractor.

This module tests the PaddingExtractor class including initialization,
forward pass, feature concatenation, and integration with utility functions.
"""

import pytest
import torch
from gymnasium import spaces
from unittest.mock import patch

from ..padding_extractor import PaddingExtractor


class TestPaddingExtractor:
    """Test suite for PaddingExtractor class."""

    def create_test_observation_space(self, include_acceleration: bool = False):
        """Create a test observation space for the extractor."""
        agent_size = 5 if include_acceleration else 3
        obstacle_size = 7 if include_acceleration else 5

        return spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(agent_size,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(10, obstacle_size),
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

    def create_test_observations(
        self,
        batch_size: int = 2,
        max_obstacles: int = 10,
        include_acceleration: bool = False,
    ):
        """Create test observation tensors."""
        agent_size = 5 if include_acceleration else 3
        obstacle_size = 7 if include_acceleration else 5

        return {
            "agent": torch.randn(batch_size, agent_size),
            "obstacles": torch.randn(batch_size, max_obstacles, obstacle_size),
            "target": torch.randn(batch_size, 2),
            "mask": torch.randint(0, 2, (batch_size, max_obstacles)).float(),
        }

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        observation_space = self.create_test_observation_space()

        extractor = PaddingExtractor(observation_space)

        assert extractor._max_obstacles == 10
        assert extractor._include_acceleration == False
        assert extractor._agent_size == 2  # velocity only
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4  # rel_pos + rel_vel
        assert extractor._obstacles_total_size == 40  # 4 * 10
        assert extractor._features_dim == 44  # 2 + 2 + 40

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        observation_space = self.create_test_observation_space()

        extractor = PaddingExtractor(
            observation_space, max_obstacles=5, include_acceleration=True
        )

        assert extractor._max_obstacles == 5
        assert extractor._include_acceleration == True
        assert extractor._agent_size == 4  # velocity + acceleration
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 6  # rel_pos + rel_vel + acc
        assert extractor._obstacles_total_size == 30  # 6 * 5
        assert extractor._features_dim == 36  # 4 + 2 + 30

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result = extractor.forward(observations)

        # Check output shape
        expected_shape = (2, extractor._features_dim)
        assert result.shape == expected_shape

        # Check that result is not all zeros (indicating proper processing)
        assert not torch.all(result == 0)

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(
            observation_space, max_obstacles=3, include_acceleration=True
        )

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        result = extractor.forward(observations)

        # Check that output includes acceleration features
        expected_features_dim = 4 + 2 + (6 * 3)  # agent + target + obstacles
        assert result.shape == (1, expected_features_dim)

    def test_forward_feature_concatenation_order(self):
        """Test that features are concatenated in the correct order."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=2)

        # Create specific test data
        observations = {
            "agent": torch.tensor(
                [[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=torch.float32
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 2.0, 3.0, 1.0, 2.0, 0.0, 0.0],
                        [0.3, 4.0, 5.0, -1.0, -2.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "target": torch.tensor([[7.0, 8.0]], dtype=torch.float32),
            "mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }

        result = extractor.forward(observations)

        # Check that we can identify the components
        agent_features = result[:, : extractor._agent_size]
        target_features = result[
            :,
            extractor._agent_size : extractor._agent_size
            + extractor._target_size,
        ]

        # Agent features should be velocity [3.0, 4.0]
        expected_agent = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        torch.testing.assert_close(agent_features, expected_agent)

        # Target features should be relative position [6.0, 6.0] (target - agent_pos)
        expected_target = torch.tensor([[6.0, 6.0]], dtype=torch.float32)
        torch.testing.assert_close(target_features, expected_target)

    def test_forward_with_masked_obstacles(self):
        """Test forward pass with some obstacles masked out."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 3, 7),
            "target": torch.randn(1, 2),
            "mask": torch.tensor(
                [[1.0, 1.0, 0.0]], dtype=torch.float32
            ),  # Last obstacle masked
        }

        result = extractor.forward(observations)

        # Extract obstacle portion of features
        obstacle_start = extractor._agent_size + extractor._target_size
        obstacle_features = result[:, obstacle_start:].reshape(1, 3, 4)

        # Last obstacle should have zero features due to masking
        torch.testing.assert_close(
            obstacle_features[:, 2, :], torch.zeros(1, 4)
        )

    def test_forward_single_batch_item(self):
        """Test forward pass with single batch item."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=5)

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=5
        )

        result = extractor.forward(observations)

        assert result.shape == (1, extractor._features_dim)

    def test_forward_large_batch(self):
        """Test forward pass with large batch size."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=4)

        batch_size = 32
        observations = self.create_test_observations(
            batch_size=batch_size, max_obstacles=4
        )

        result = extractor.forward(observations)

        assert result.shape == (batch_size, extractor._features_dim)

    def test_forward_zero_obstacles(self):
        """Test forward pass with no valid obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 3, 7),
            "target": torch.randn(1, 2),
            "mask": torch.zeros(1, 3),  # All obstacles masked
        }

        result = extractor.forward(observations)

        # Should still produce valid output
        assert result.shape == (1, extractor._features_dim)

        # Obstacle portion should be zeros
        obstacle_start = extractor._agent_size + extractor._target_size
        obstacle_features = result[:, obstacle_start:]
        torch.testing.assert_close(
            obstacle_features, torch.zeros(1, extractor._obstacles_total_size)
        )

    def test_forward_deterministic(self):
        """Test that forward pass is deterministic with same input."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result1 = extractor.forward(observations)
        result2 = extractor.forward(observations)

        torch.testing.assert_close(result1, result2)

    @patch("extractors.padding_extractor.validate_observation_tensors")
    def test_forward_calls_validation(self, mock_validate):
        """Test that forward pass calls validation function."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        extractor.forward(observations)

        # Check that validation was called
        mock_validate.assert_called_once()

        # Check validation was called with correct arguments
        args, kwargs = mock_validate.call_args
        assert args[0] is observations["agent"]
        assert args[1] is observations["obstacles"]
        assert args[2] is observations["target"]
        assert args[3] is observations["mask"]
        assert args[4] == 3  # max_obstacles

    def test_different_obstacle_counts(self):
        """Test extractor with different numbers of obstacles."""
        observation_space = self.create_test_observation_space()

        for max_obstacles in [1, 5, 15, 20]:
            extractor = PaddingExtractor(
                observation_space, max_obstacles=max_obstacles
            )
            observations = self.create_test_observations(
                batch_size=1, max_obstacles=max_obstacles
            )

            result = extractor.forward(observations)

            expected_dim = (
                2 + 2 + (4 * max_obstacles)
            )  # agent + target + obstacles
            assert result.shape == (1, expected_dim)

    def test_features_dim_property(self):
        """Test that features_dim property returns correct value."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(
            observation_space, max_obstacles=7, include_acceleration=True
        )

        expected_dim = 4 + 2 + (6 * 7)  # agent(vel+acc) + target + obstacles
        assert extractor.features_dim == expected_dim
        assert extractor._features_dim == expected_dim

    def test_reproducibility_with_torch_manual_seed(self):
        """Test reproducibility when using torch manual seed."""
        observation_space = self.create_test_observation_space()

        # First run
        torch.manual_seed(42)
        extractor1 = PaddingExtractor(observation_space, max_obstacles=3)
        observations1 = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )
        result1 = extractor1.forward(observations1)

        # Second run with same seed
        torch.manual_seed(42)
        extractor2 = PaddingExtractor(observation_space, max_obstacles=3)
        observations2 = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )
        result2 = extractor2.forward(observations2)

        # Results should be identical (within floating point precision)
        torch.testing.assert_close(result1, result2, rtol=1e-5, atol=1e-6)


class TestPaddingExtractorIntegration:
    """Integration tests for PaddingExtractor with realistic scenarios."""

    def test_realistic_navigation_scenario(self):
        """Test with realistic navigation scenario data."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(5, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(5,)),
            }
        )

        extractor = PaddingExtractor(observation_space, max_obstacles=5)

        # Realistic scenario: agent moving towards target with some obstacles
        observations = {
            "agent": torch.tensor(
                [[0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),  # Moving right
            "obstacles": torch.tensor(
                [
                    [
                        [
                            0.3,
                            2.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],  # Static obstacle ahead
                        [0.3, 1.0, 1.0, -0.5, 0.0, 0.0, 0.0],  # Moving obstacle
                        [
                            0.3,
                            -1.0,
                            -1.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],  # Static obstacle behind
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Invalid obstacle
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),  # Invalid obstacle
            "target": torch.tensor(
                [[5.0, 0.0]], dtype=torch.float32
            ),  # Target to the right
            "mask": torch.tensor(
                [[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32
            ),  # 3 valid obstacles
        }

        result = extractor.forward(observations)

        # Should produce reasonable output
        assert result.shape == (1, extractor._features_dim)
        assert torch.all(torch.isfinite(result))
        assert not torch.all(result == 0)

    def test_edge_case_same_positions(self):
        """Test edge case where agent, target, and obstacles have same positions."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(2,)),
            }
        )

        extractor = PaddingExtractor(observation_space, max_obstacles=2)

        # Everything at same position (0, 0)
        observations = {
            "agent": torch.tensor(
                [[0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0],
                        [0.3, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "target": torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            "mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        }

        result = extractor.forward(observations)

        # Should handle this gracefully
        assert result.shape == (1, extractor._features_dim)
        assert torch.all(torch.isfinite(result))

    def test_performance_large_batch_many_obstacles(self):
        """Test performance with large batch and many obstacles."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(20, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(20,)),
            }
        )

        extractor = PaddingExtractor(observation_space, max_obstacles=20)

        batch_size = 64
        observations = {
            "agent": torch.randn(batch_size, 7),
            "obstacles": torch.randn(batch_size, 20, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, 20),
        }

        # Should complete without issues
        result = extractor.forward(observations)

        assert result.shape == (batch_size, extractor._features_dim)
        assert torch.all(torch.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
