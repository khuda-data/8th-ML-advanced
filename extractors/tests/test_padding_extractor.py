"""Comprehensive tests for PaddingExtractor.
This module tests the PaddingExtractor class including initialization,
forward pass, feature concatenation, and padding mechanism.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch

from ..padding_extractor import PaddingExtractor


class TestPaddingExtractor:
    """Test suite for PaddingExtractor class."""

    def create_test_observation_space(
        self, include_acceleration: bool = False, include_radius: bool = True
    ):
        """Create a test observation space for the extractor."""
        # Calculate feature sizes based on flags
        agent_size = 2  # Base: vel_x, vel_y
        if include_radius:
            agent_size += 1  # Add radius
        if include_acceleration:
            agent_size += 2  # Add acc_x, acc_y

        obstacle_size = 4  # Base: rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y
        if include_radius:
            obstacle_size += 1  # Add radius
        if include_acceleration:
            obstacle_size += 2  # Add acc_x, acc_y

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
        include_radius: bool = True,
    ):
        """Create test observation tensors."""
        # Calculate feature sizes based on flags
        agent_size = 2  # Base: vel_x, vel_y
        if include_radius:
            agent_size += 1  # Add radius
        if include_acceleration:
            agent_size += 2  # Add acc_x, acc_y

        obstacle_size = 4  # Base: rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y
        if include_radius:
            obstacle_size += 1  # Add radius
        if include_acceleration:
            obstacle_size += 2  # Add acc_x, acc_y

        return {
            "agent": torch.randn(batch_size, agent_size),
            "obstacles": torch.randn(batch_size, max_obstacles, obstacle_size),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, max_obstacles),
        }

    def test_initialization_default_parameters(self):
        """Test PaddingExtractor initialization with default parameters."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        # Test default parameter values
        assert extractor.include_acceleration is False
        assert extractor.include_radius is True

        # Test feature dimensions calculation
        expected_agent_features = 3  # radius + vel_x + vel_y
        expected_obstacle_features = 5  # radius + rel_pos + rel_vel
        expected_max_obstacles = 10
        expected_features_dim = (
            expected_agent_features
            + 2
            + (expected_max_obstacles * expected_obstacle_features)
        )

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.max_obstacles == expected_max_obstacles
        assert extractor.features_dim == expected_features_dim

    def test_initialization_custom_parameters(self):
        """Test PaddingExtractor initialization with custom parameters."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=False
        )
        extractor = PaddingExtractor(
            observation_space,
            max_obstacles=10,
            include_acceleration=True,
            include_radius=False,
        )

        # Test custom parameter values
        assert extractor.include_acceleration is True
        assert extractor.include_radius is False

        # Test feature dimensions calculation
        expected_agent_features = 4  # vel_x + vel_y + acc_x + acc_y (no radius)
        expected_obstacle_features = 6  # rel_pos + rel_vel + acc (no radius)
        expected_max_obstacles = 10
        expected_features_dim = (
            expected_agent_features
            + 2
            + (expected_max_obstacles * expected_obstacle_features)
        )

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.features_dim == expected_features_dim

    def test_forward_basic(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)
        observations = self.create_test_observations()

        with torch.no_grad():
            features = extractor(observations)

        # agent(3) + target(2) + obstacles(10*5)
        expected_features_dim = 3 + 2 + (10 * 5)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=True
        )
        extractor = PaddingExtractor(
            observation_space,
            max_obstacles=10,
            include_acceleration=True,
            include_radius=True,
        )
        observations = self.create_test_observations(
            include_acceleration=True, include_radius=True
        )

        with torch.no_grad():
            features = extractor(observations)

        # agent(5) + target(2) + obstacles(10*7)
        expected_features_dim = 5 + 2 + (10 * 7)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_without_radius(self):
        """Test forward pass without radius features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=False, include_radius=False
        )
        extractor = PaddingExtractor(
            observation_space, include_acceleration=False, include_radius=False
        )
        observations = self.create_test_observations(
            include_acceleration=False, include_radius=False
        )

        with torch.no_grad():
            features = extractor(observations)

        # agent(2) + target(2) + obstacles(10*4)
        expected_features_dim = 2 + 2 + (10 * 4)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_feature_concatenation_order(self):
        """Test that features are concatenated in the correct order."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        # Create predictable test data
        batch_size = 1
        max_obstacles = 3

        observations = {
            "agent": torch.tensor([[1.0, 2.0, 3.0]]),  # [radius, vel_x, vel_y]
            "obstacles": torch.tensor(
                [
                    [
                        [4.0, 5.0, 6.0, 7.0, 8.0],  # obstacle 1
                        [9.0, 10.0, 11.0, 12.0, 13.0],  # obstacle 2
                        [14.0, 15.0, 16.0, 17.0, 18.0],
                    ]
                ]
            ),  # obstacle 3
            "target": torch.tensor([[19.0, 20.0]]),
            "mask": torch.ones(batch_size, max_obstacles),
        }

        with torch.no_grad():
            features = extractor(observations)

        # Check concatenation order: agent + target + flattened obstacles
        expected = torch.cat(
            [
                observations["agent"].flatten(),  # [1, 2, 3]
                observations["target"].flatten(),  # [19, 20]
                observations[
                    "obstacles"
                ].flatten(),  # [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
            ]
        )

        torch.testing.assert_close(features[0, : len(expected)], expected)

    def test_forward_with_masked_obstacles(self):
        """Test that masked obstacles are properly zeroed."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=4
        )
        observations["mask"] = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0]]
        )  # Only first 2 obstacles valid

        with torch.no_grad():
            features = extractor(observations)

        # Extract obstacle portion of features (after agent + target)
        agent_target_size = 3 + 2  # agent + target
        obstacle_features = features[0, agent_target_size:].reshape(4, 5)

        # First two obstacles should retain values, last two should be zero
        assert not torch.allclose(
            obstacle_features[0], torch.zeros(5)
        )  # First obstacle has values
        assert not torch.allclose(
            obstacle_features[1], torch.zeros(5)
        )  # Second obstacle has values
        torch.testing.assert_close(
            obstacle_features[2], torch.zeros(5)
        )  # Third obstacle is zero
        torch.testing.assert_close(
            obstacle_features[3], torch.zeros(5)
        )  # Fourth obstacle is zero

    def test_forward_zero_obstacles(self):
        """Test forward pass when all obstacles are masked."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=1)
        observations["mask"] = torch.zeros(1, 10)  # All obstacles masked

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (1, 3 + 2 + (10 * 5))
        assert not torch.isnan(features).any()

        # Check that obstacle features are all zeros
        agent_target_size = 3 + 2
        obstacle_features = features[0, agent_target_size:]
        torch.testing.assert_close(obstacle_features, torch.zeros(10 * 5))

    def test_different_batch_sizes(self):
        """Test extractor with different batch sizes."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        for batch_size in [1, 4, 8, 16]:
            observations = self.create_test_observations(batch_size=batch_size)

            with torch.no_grad():
                features = extractor(observations)

            expected_shape = (batch_size, 3 + 2 + (10 * 5))
            assert features.shape == expected_shape
            assert not torch.isnan(features).any()

    def test_different_obstacle_counts(self):
        """Test extractor with different numbers of obstacles."""
        for max_obstacles in [5, 15, 20]:
            observation_space = self.create_test_observation_space()
            observation_space["obstacles"] = spaces.Box(
                low=-float("inf"), high=float("inf"), shape=(max_obstacles, 5)
            )
            observation_space["mask"] = spaces.Box(
                low=0, high=1, shape=(max_obstacles,)
            )

            extractor = PaddingExtractor(observation_space, max_obstacles=10)

            observations = self.create_test_observations(
                max_obstacles=max_obstacles
            )
            observations["obstacles"] = torch.randn(2, max_obstacles, 5)
            observations["mask"] = torch.rand(2, max_obstacles)

            with torch.no_grad():
                features = extractor(observations)

            expected_shape = (2, 3 + 2 + (max_obstacles * 5))
            assert features.shape == expected_shape
            assert not torch.isnan(features).any()

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=2)

        # Enable gradients for input tensors
        for key in observations:
            observations[key].requires_grad_(True)

        features = extractor(observations)
        loss = features.sum()
        loss.backward()

        # Check that input gradients exist
        for key in ["agent", "obstacles", "target"]:
            assert observations[key].grad is not None, f"No gradient for {key}"
            assert not torch.isnan(
                observations[key].grad
            ).any(), f"NaN gradient for {key}"

    def test_features_dim_property(self):
        """Test the features_dim property calculation."""
        # Test with different configurations
        configs = [
            (
                False,
                False,
                10,
                2 + 2 + (10 * 4),
            ),  # vel only + target + obstacles
            (
                False,
                True,
                10,
                3 + 2 + (10 * 5),
            ),  # radius + vel + target + obstacles
            (
                True,
                False,
                10,
                4 + 2 + (10 * 6),
            ),  # vel + acc + target + obstacles
            (
                True,
                True,
                10,
                5 + 2 + (10 * 7),
            ),  # radius + vel + acc + target + obstacles
            (False, True, 5, 3 + 2 + (5 * 5)),  # Test different obstacle count
        ]

        for (
            include_acceleration,
            include_radius,
            max_obs,
            expected_dim,
        ) in configs:
            observation_space = self.create_test_observation_space(
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            if max_obs != 10:
                # Update obstacle count in observation space
                agent_size = observation_space["agent"].shape[0]
                obstacle_size = observation_space["obstacles"].shape[1]
                observation_space = spaces.Dict(
                    {
                        "agent": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(agent_size,),
                        ),
                        "obstacles": spaces.Box(
                            low=-float("inf"),
                            high=float("inf"),
                            shape=(max_obs, obstacle_size),
                        ),
                        "target": spaces.Box(
                            low=-float("inf"), high=float("inf"), shape=(2,)
                        ),
                        "mask": spaces.Box(low=0, high=1, shape=(max_obs,)),
                    }
                )

            extractor = PaddingExtractor(
                observation_space,
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            assert extractor.features_dim == expected_dim


class TestPaddingExtractorIntegration:
    """Integration tests for PaddingExtractor."""

    def create_test_observation_space(
        self, include_acceleration: bool = False, include_radius: bool = True
    ):
        """Create a test observation space for integration tests."""
        # Calculate feature sizes based on flags
        agent_size = 2  # Base: vel_x, vel_y
        if include_radius:
            agent_size += 1  # Add radius
        if include_acceleration:
            agent_size += 2  # Add acc_x, acc_y

        obstacle_size = 4  # Base: rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y
        if include_radius:
            obstacle_size += 1  # Add radius
        if include_acceleration:
            obstacle_size += 2  # Add acc_x, acc_y

        return spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(agent_size,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(15, obstacle_size),
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(15,)),
            }
        )

    def test_realistic_navigation_scenario(self):
        """Test with realistic navigation scenario data."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        # Simulate realistic agent state
        agent_data = torch.tensor([[0.5, 0.3, 0.1]])  # [radius, vel_x, vel_y]

        # Simulate realistic obstacles
        obstacles_data = torch.zeros(1, 15, 5)
        obstacles_data[0, 0, :] = torch.tensor(
            [0.3, 0.5, 0.2, -0.1, 0.0]
        )  # Close obstacle
        obstacles_data[0, 1, :] = torch.tensor(
            [0.4, 2.0, 1.5, 0.0, 0.1]
        )  # Distant obstacle
        obstacles_data[0, 2, :] = torch.tensor(
            [0.2, -0.8, 0.3, 0.2, -0.1]
        )  # Another close obstacle

        # Target position
        target_data = torch.tensor([[1.5, 2.0]])

        # Mask - first 3 obstacles are valid
        mask = torch.zeros(1, 15)
        mask[0, :3] = 1.0

        observations = {
            "agent": agent_data,
            "obstacles": obstacles_data,
            "target": target_data,
            "mask": mask,
        }

        with torch.no_grad():
            features = extractor(observations)

        expected_dim = 3 + 2 + (15 * 5)  # 80
        assert features.shape == (1, expected_dim)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()

    def test_edge_case_same_positions(self):
        """Test edge case where agent and obstacles have same positions."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        # Agent and obstacles at same position (zero relative distance)
        observations = {
            "agent": torch.tensor([[0.5, 0.0, 0.0]]),  # [radius, vel_x, vel_y]
            "obstacles": torch.tensor(
                [[[0.3, 0.0, 0.0, 0.0, 0.0]]]
            ),  # Same position
            "target": torch.tensor(
                [[0.0, 0.0]]
            ),  # Target also at same position
            "mask": torch.tensor([[1.0]]),
        }

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape[0] == 1
        assert not torch.isnan(features).any()

    def test_performance_large_batch_many_obstacles(self):
        """Test extractor performance with large batch and many obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        batch_size = 64
        max_obstacles = 15

        observations = {
            "agent": torch.randn(batch_size, 3),
            "obstacles": torch.randn(batch_size, max_obstacles, 5),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, max_obstacles),
        }

        with torch.no_grad():
            features = extractor(observations)

        expected_dim = 3 + 2 + (15 * 5)
        assert features.shape == (batch_size, expected_dim)
        assert not torch.isnan(features).any()

    def test_consistency_across_runs(self):
        """Test that the extractor produces consistent results."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        observations = {
            "agent": torch.tensor([[0.5, 0.1, 0.2]]),
            "obstacles": torch.tensor(
                [[[0.3, 1.0, 0.5, 0.1, 0.0], [0.4, 0.5, 1.0, -0.1, 0.1]]]
            ),
            "target": torch.tensor([[1.5, 1.0]]),
            "mask": torch.tensor([[1.0, 1.0]]),
        }

        # Run multiple times
        results = []
        for _ in range(5):
            with torch.no_grad():
                features = extractor(observations)
            results.append(features.clone())

        # All results should be identical (deterministic)
        for result in results[1:]:
            torch.testing.assert_close(results[0], result)

    def test_memory_efficiency(self):
        """Test that the extractor doesn't accumulate excessive memory."""
        observation_space = self.create_test_observation_space()
        extractor = PaddingExtractor(observation_space, max_obstacles=10)

        # Run multiple forward passes to check for memory leaks
        for _ in range(10):
            observations = {
                "agent": torch.randn(8, 3),
                "obstacles": torch.randn(8, 15, 5),
                "target": torch.randn(8, 2),
                "mask": torch.rand(8, 15),
            }

            with torch.no_grad():
                features = extractor(observations)

            del features, observations

        # Test passes if no memory error occurs


if __name__ == "__main__":
    pytest.main([__file__])
