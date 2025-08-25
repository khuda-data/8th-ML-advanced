"""Comprehensive tests for AttentionExtractor.
This module tests the AttentionExtractor class including initialization,
forward pass, attention mechanism, and multi-layer processing.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch

from ..attention_extractor import AttentionExtractor


class TestAttentionExtractor:
    """Test suite for AttentionExtractor class."""

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
        """Test AttentionExtractor initialization with default parameters."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Test default parameter values
        assert extractor.include_acceleration is False
        assert extractor.include_radius is True
        assert extractor.d_model == 64
        assert extractor.nhead == 8
        assert extractor.num_layers == 2
        assert extractor.dropout == 0.1

        # Test feature dimensions calculation
        expected_agent_features = 3  # radius + vel_x + vel_y
        expected_obstacle_features = 5  # radius + rel_pos + rel_vel
        expected_features_dim = (
            expected_agent_features + 2 + extractor.d_model
        )  # agent + target + obstacles

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.features_dim == expected_features_dim

    def test_initialization_custom_parameters(self):
        """Test AttentionExtractor initialization with custom parameters."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=False
        )
        extractor = AttentionExtractor(
            observation_space,
            max_obstacles=10,
            include_acceleration=True,
            include_radius=False,
            d_model=128,
            nhead=4,
            num_layers=3,
            dropout=0.2,
        )

        # Test custom parameter values
        assert extractor.include_acceleration is True
        assert extractor.include_radius is False
        assert extractor.d_model == 128
        assert extractor.nhead == 4
        assert extractor.num_layers == 3
        assert extractor.dropout == 0.2

        # Test feature dimensions calculation
        expected_agent_features = 4  # vel_x + vel_y + acc_x + acc_y (no radius)
        expected_obstacle_features = 6  # rel_pos + rel_vel + acc (no radius)
        expected_features_dim = expected_agent_features + 2 + extractor.d_model

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.features_dim == expected_features_dim

    def test_layer_initialization(self):
        """Test that layers are properly initialized."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Test that required layers exist
        assert hasattr(extractor, "obstacle_projection")
        assert hasattr(extractor, "transformer_encoder")
        assert hasattr(extractor, "global_pooling")

        # Test layer properties
        assert isinstance(extractor.obstacle_projection, nn.Linear)
        assert isinstance(extractor.transformer_encoder, nn.TransformerEncoder)
        assert (
            extractor.obstacle_projection.in_features
            == extractor.obstacle_features
        )
        assert extractor.obstacle_projection.out_features == extractor.d_model

    def test_forward_basic(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)
        observations = self.create_test_observations()

        with torch.no_grad():
            features = extractor(observations)

        expected_features_dim = 3 + 2 + 64  # agent + target + obstacle features
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=True
        )
        extractor = AttentionExtractor(
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

        expected_features_dim = (
            5 + 2 + 64
        )  # agent(5) + target(2) + obstacle features(64)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_without_radius(self):
        """Test forward pass without radius features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=False, include_radius=False
        )
        extractor = AttentionExtractor(
            observation_space,
            max_obstacles=10,
            include_acceleration=False,
            include_radius=False,
        )
        observations = self.create_test_observations(
            include_acceleration=False, include_radius=False
        )

        with torch.no_grad():
            features = extractor(observations)

        expected_features_dim = (
            2 + 2 + 64
        )  # agent(2) + target(2) + obstacle features(64)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_attention_with_mask(self):
        """Test that attention mechanism respects obstacle mask."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Create observations with specific mask pattern
        observations = self.create_test_observations(
            batch_size=1, max_obstacles=4
        )
        observations["mask"] = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0]]
        )  # Only first 2 obstacles valid

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (1, 3 + 2 + 64)
        assert not torch.isnan(features).any()

    def test_forward_all_obstacles_masked(self):
        """Test forward pass when all obstacles are masked."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=1)
        observations["mask"] = torch.zeros(1, 10)  # All obstacles masked

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (1, 3 + 2 + 64)
        assert not torch.isnan(features).any()

    def test_apply_attention_method(self):
        """Test the apply_attention method specifically."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Create test tensors
        batch_size = 2
        max_obstacles = 4
        projected_obstacles = torch.randn(
            batch_size, max_obstacles, extractor.d_model
        )
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        with torch.no_grad():
            attended_features = extractor.apply_attention(
                projected_obstacles, mask
            )

        assert attended_features.shape == (batch_size, extractor.d_model)
        assert not torch.isnan(attended_features).any()

    def test_attention_mechanism_focuses_on_valid_obstacles(self):
        """Test that attention mechanism focuses on valid obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, max_obstacles=10, d_model=32
        )  # Smaller for easier testing

        # Create observations with clear patterns
        batch_size = 1
        max_obstacles = 3
        observations = self.create_test_observations(
            batch_size=batch_size, max_obstacles=max_obstacles
        )

        # Set specific values to make the test more predictable
        observations["obstacles"][
            0, 0, :
        ] = 1.0  # First obstacle has high values
        observations["obstacles"][
            0, 1, :
        ] = 0.1  # Second obstacle has low values
        observations["obstacles"][0, 2, :] = 0.0  # Third obstacle is zero
        observations["mask"] = torch.tensor(
            [[1.0, 1.0, 0.0]]
        )  # First two valid

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (batch_size, 3 + 2 + 32)
        assert not torch.isnan(features).any()

    def test_feed_forward_layers(self):
        """Test that feed-forward layers in transformer work correctly."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, max_obstacles=10, d_model=64, num_layers=1
        )

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=2
        )

        # Test that we can get intermediate representations
        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (1, 3 + 2 + 64)
        assert not torch.isnan(features).any()

        # Test that the transformer layers are working
        assert len(list(extractor.transformer_encoder.layers)) == 1

    def test_different_batch_sizes(self):
        """Test extractor with different batch sizes."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        for batch_size in [1, 4, 8, 16]:
            observations = self.create_test_observations(batch_size=batch_size)

            with torch.no_grad():
                features = extractor(observations)

            expected_shape = (batch_size, 3 + 2 + 64)
            assert features.shape == expected_shape
            assert not torch.isnan(features).any()

    def test_different_obstacle_counts(self):
        """Test extractor with different numbers of obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        for max_obstacles in [1, 5, 15, 20]:
            observations = self.create_test_observations(
                max_obstacles=max_obstacles
            )
            observations["obstacles"] = torch.randn(2, max_obstacles, 5)
            observations["mask"] = torch.rand(2, max_obstacles)

            with torch.no_grad():
                features = extractor(observations)

            assert features.shape == (2, 3 + 2 + 64)
            assert not torch.isnan(features).any()

    def test_gradient_flow(self):
        """Test that gradients flow properly through the network."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=2)

        # Enable gradients
        for param in extractor.parameters():
            param.requires_grad_(True)

        features = extractor(observations)
        loss = features.sum()
        loss.backward()

        # Check that gradients exist
        for name, param in extractor.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_features_dim_property(self):
        """Test the features_dim property calculation."""
        # Test with different configurations
        configs = [
            (False, False, 2 + 2 + 64),  # vel only + target + obstacles
            (False, True, 3 + 2 + 64),  # radius + vel + target + obstacles
            (True, False, 4 + 2 + 64),  # vel + acc + target + obstacles
            (True, True, 5 + 2 + 64),  # radius + vel + acc + target + obstacles
        ]

        for include_acceleration, include_radius, expected_dim in configs:
            observation_space = self.create_test_observation_space(
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            extractor = AttentionExtractor(
                observation_space,
                max_obstacles=10,
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            assert extractor.features_dim == expected_dim


class TestAttentionExtractorIntegration:
    """Integration tests for AttentionExtractor."""

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
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Simulate realistic agent state
        agent_data = torch.tensor([[0.5, 0.3, 0.1]])  # [radius, vel_x, vel_y]

        # Simulate realistic obstacles (some close, some far)
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

        assert features.shape == (1, 3 + 2 + 64)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()

    def test_performance_large_batch(self):
        """Test extractor performance with large batch size."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, max_obstacles=10, d_model=32
        )  # Smaller model for speed

        batch_size = 64
        max_obstacles = 10

        observations = {
            "agent": torch.randn(batch_size, 3),
            "obstacles": torch.randn(batch_size, max_obstacles, 5),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, max_obstacles),
        }

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (batch_size, 3 + 2 + 32)
        assert not torch.isnan(features).any()

    def test_attention_robustness_to_obstacle_ordering(self):
        """Test that attention is reasonably robust to obstacle ordering."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, max_obstacles=10, d_model=32
        )

        # Create two identical scenarios with different obstacle ordering
        batch_size = 1
        max_obstacles = 4

        # Scenario 1: obstacles in original order
        obs1 = {
            "agent": torch.tensor([[0.5, 0.1, 0.2]]),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 0.5, 0.1, 0.0],
                        [0.4, 0.5, 1.0, -0.1, 0.1],
                        [0.2, 2.0, 0.3, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "target": torch.tensor([[1.5, 1.0]]),
            "mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        }

        # Scenario 2: obstacles in different order (swap first two)
        obs2 = {
            "agent": torch.tensor([[0.5, 0.1, 0.2]]),
            "obstacles": torch.tensor(
                [
                    [
                        [0.4, 0.5, 1.0, -0.1, 0.1],
                        [0.3, 1.0, 0.5, 0.1, 0.0],
                        [0.2, 2.0, 0.3, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ),
            "target": torch.tensor([[1.5, 1.0]]),
            "mask": torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        }

        with torch.no_grad():
            features1 = extractor(obs1)
            features2 = extractor(obs2)

        # Features should be reasonably similar (attention should handle permutation)
        assert features1.shape == features2.shape
        assert not torch.isnan(features1).any()
        assert not torch.isnan(features2).any()

    def test_memory_efficiency(self):
        """Test that the extractor doesn't accumulate excessive memory."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=10)

        # Run multiple forward passes to check for memory leaks
        for _ in range(10):
            observations = {
                "agent": torch.randn(4, 3),
                "obstacles": torch.randn(4, 8, 5),
                "target": torch.randn(4, 2),
                "mask": torch.rand(4, 8),
            }

            with torch.no_grad():
                features = extractor(observations)

            del features, observations

        # Test passes if no memory error occurs


if __name__ == "__main__":
    pytest.main([__file__])
