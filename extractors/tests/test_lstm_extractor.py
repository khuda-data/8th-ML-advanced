"""Comprehensive tests for LSTMExtractor.
This module tests the LSTMExtractor class including initialization,
forward pass, LSTM sequence processing, and feature extraction.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch

from ..lstm_extractor import LSTMExtractor


class TestLSTMExtractor:
    """Test suite for LSTMExtractor class."""

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
        """Test LSTMExtractor initialization with default parameters."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        # Test default parameter values
        assert extractor.include_acceleration is False
        assert extractor.include_radius is True
        assert extractor.hidden_size == 64
        assert extractor.num_layers == 2
        assert extractor.dropout == 0.1

        # Test feature dimensions calculation
        expected_agent_features = 3  # radius + vel_x + vel_y
        expected_obstacle_features = 5  # radius + rel_pos + rel_vel
        expected_features_dim = (
            expected_agent_features + 2 + extractor.hidden_size
        )

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.features_dim == expected_features_dim

    def test_initialization_custom_parameters(self):
        """Test LSTMExtractor initialization with custom parameters."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=False
        )
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=10,
            include_acceleration=True,
            include_radius=False,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
        )

        # Test custom parameter values
        assert extractor.include_acceleration is True
        assert extractor.include_radius is False
        assert extractor.hidden_size == 128
        assert extractor.num_layers == 3
        assert extractor.dropout == 0.2

        # Test feature dimensions calculation
        expected_agent_features = 4  # vel_x + vel_y + acc_x + acc_y (no radius)
        expected_obstacle_features = 6  # rel_pos + rel_vel + acc (no radius)
        expected_features_dim = (
            expected_agent_features + 2 + extractor.hidden_size
        )

        assert extractor.agent_features == expected_agent_features
        assert extractor.obstacle_features == expected_obstacle_features
        assert extractor.features_dim == expected_features_dim

    def test_layer_initialization(self):
        """Test that LSTM layers are properly initialized."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        # Test that required layers exist
        assert hasattr(extractor, "lstm")
        assert hasattr(extractor, "dropout_layer")

        # Test layer properties
        assert isinstance(extractor.lstm, nn.LSTM)
        assert isinstance(extractor.dropout_layer, nn.Dropout)
        assert extractor.lstm.input_size == extractor.obstacle_features
        assert extractor.lstm.hidden_size == extractor.hidden_size
        assert extractor.lstm.num_layers == extractor.num_layers

    def test_forward_basic(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)
        observations = self.create_test_observations()

        with torch.no_grad():
            features = extractor(observations)

        expected_features_dim = 3 + 2 + 64  # agent + target + LSTM output
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=True, include_radius=True
        )
        extractor = LSTMExtractor(
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
        )  # agent(5) + target(2) + LSTM output(64)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_without_radius(self):
        """Test forward pass without radius features."""
        observation_space = self.create_test_observation_space(
            include_acceleration=False, include_radius=False
        )
        extractor = LSTMExtractor(
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
        )  # agent(2) + target(2) + LSTM output(64)
        assert features.shape == (2, expected_features_dim)
        assert not torch.isnan(features).any()

    def test_forward_with_mask(self):
        """Test that LSTM respects obstacle mask."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

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
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=1)
        observations["mask"] = torch.zeros(1, 10)  # All obstacles masked

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (1, 3 + 2 + 64)
        assert not torch.isnan(features).any()

    def test_lstm_sequence_processing(self):
        """Test that LSTM properly processes obstacle sequences."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=10, hidden_size=32
        )

        # Create a predictable sequence
        batch_size = 1
        max_obstacles = 5
        observations = self.create_test_observations(
            batch_size=batch_size, max_obstacles=max_obstacles
        )

        # Set obstacles with increasing values to test sequence processing
        for i in range(max_obstacles):
            observations["obstacles"][0, i, :] = float(i + 1) * torch.ones(5)
        observations["mask"] = torch.ones(batch_size, max_obstacles)

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (batch_size, 3 + 2 + 32)
        assert not torch.isnan(features).any()

    def test_different_batch_sizes(self):
        """Test extractor with different batch sizes."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

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
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

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
        """Test that gradients flow properly through the LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

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

    def test_hidden_state_reset(self):
        """Test that hidden states are properly managed."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        observations = self.create_test_observations(batch_size=2)

        # Run multiple forward passes
        with torch.no_grad():
            features1 = extractor(observations)
            features2 = extractor(observations)

        # Should produce consistent results (no hidden state carryover)
        assert features1.shape == features2.shape
        assert not torch.isnan(features1).any()
        assert not torch.isnan(features2).any()

    def test_features_dim_property(self):
        """Test the features_dim property calculation."""
        # Test with different configurations
        configs = [
            (False, False, 2 + 2 + 64),  # vel only + target + LSTM
            (False, True, 3 + 2 + 64),  # radius + vel + target + LSTM
            (True, False, 4 + 2 + 64),  # vel + acc + target + LSTM
            (True, True, 5 + 2 + 64),  # radius + vel + acc + target + LSTM
        ]

        for include_acceleration, include_radius, expected_dim in configs:
            observation_space = self.create_test_observation_space(
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            extractor = LSTMExtractor(
                observation_space,
                max_obstacles=10,
                include_acceleration=include_acceleration,
                include_radius=include_radius,
            )
            assert extractor.features_dim == expected_dim


class TestLSTMExtractorIntegration:
    """Integration tests for LSTMExtractor."""

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
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        # Simulate realistic agent state
        agent_data = torch.tensor([[0.5, 0.3, 0.1]])  # [radius, vel_x, vel_y]

        # Simulate realistic obstacles with temporal patterns
        obstacles_data = torch.zeros(2, 15, 5)

        # Batch 1: obstacles moving towards agent
        obstacles_data[0, 0, :] = torch.tensor(
            [0.3, 0.5, 0.2, -0.1, 0.0]
        )  # Close, approaching
        obstacles_data[0, 1, :] = torch.tensor(
            [0.4, 2.0, 1.5, -0.2, -0.1]
        )  # Distant, approaching
        obstacles_data[0, 2, :] = torch.tensor(
            [0.2, -0.8, 0.3, 0.1, -0.05]
        )  # Behind, moving away

        # Batch 2: different scenario
        obstacles_data[1, 0, :] = torch.tensor(
            [0.25, 1.0, 0.8, 0.0, 0.1]
        )  # Stationary
        obstacles_data[1, 1, :] = torch.tensor(
            [0.35, 0.3, -0.5, 0.15, 0.0]
        )  # Close, crossing

        # Target positions
        target_data = torch.tensor([[1.5, 2.0], [0.8, 1.2]])

        # Masks - different valid obstacle counts
        mask = torch.zeros(2, 15)
        mask[0, :3] = 1.0  # First batch has 3 valid obstacles
        mask[1, :2] = 1.0  # Second batch has 2 valid obstacles

        observations = {
            "agent": torch.tensor([[0.5, 0.3, 0.1], [0.4, 0.2, 0.05]]),
            "obstacles": obstacles_data,
            "target": target_data,
            "mask": mask,
        }

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (2, 3 + 2 + 64)
        assert not torch.isnan(features).any()
        assert torch.isfinite(features).all()

    def test_performance_large_batch(self):
        """Test extractor performance with large batch size."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=10, hidden_size=32
        )  # Smaller for speed

        batch_size = 32
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

    def test_edge_case_single_obstacle(self):
        """Test with single obstacle scenarios."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

        batch_size = 2
        observations = {
            "agent": torch.tensor([[0.5, 0.1, 0.2], [0.4, 0.3, 0.1]]),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 0.5, 0.1, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],  # Batch 1: 1 obstacle
                    [
                        [0.4, 0.8, -0.3, -0.1, 0.1],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],  # Batch 2: 1 obstacle
                ]
            ),
            "target": torch.tensor([[1.5, 1.0], [0.5, 1.2]]),
            "mask": torch.tensor(
                [[1.0, 0.0], [1.0, 0.0]]
            ),  # Only first obstacle valid
        }

        with torch.no_grad():
            features = extractor(observations)

        assert features.shape == (batch_size, 3 + 2 + 64)
        assert not torch.isnan(features).any()

    def test_sequence_length_sensitivity(self):
        """Test LSTM sensitivity to sequence length."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=10, hidden_size=32
        )

        # Test with different numbers of valid obstacles
        for num_valid in [1, 3, 5, 8]:
            observations = {
                "agent": torch.randn(1, 3),
                "obstacles": torch.randn(1, 10, 5),
                "target": torch.randn(1, 2),
                "mask": torch.zeros(1, 10),
            }
            # Set first num_valid obstacles as valid
            observations["mask"][0, :num_valid] = 1.0

            with torch.no_grad():
                features = extractor(observations)

            assert features.shape == (1, 3 + 2 + 32)
            assert not torch.isnan(features).any()

    def test_memory_efficiency(self):
        """Test that the extractor doesn't accumulate excessive memory."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=10)

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
