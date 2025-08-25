"""Comprehensive tests for AttentionExtractor.

This module tests the AttentionExtractor functionality including initialization,
forward pass, attention mechanism, and feature extraction with proper batch dimensions.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch, MagicMock

from ..attention_extractor import AttentionExtractor


class TestAttentionExtractorInit:
    """Test suite for AttentionExtractor initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(10, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

        extractor = AttentionExtractor(observation_space)

        # Test default values
        assert extractor._d_model == 64
        assert extractor._num_heads == 4
        assert extractor._num_layers == 1
        assert extractor._max_obstacles == 10
        assert (
            extractor._include_acceleration == True
        )  # Corrected default value
        assert extractor._include_radius == True
        assert extractor._agent_size == 5  # vel_x, vel_y, acc_x, acc_y, radius
        assert extractor._target_size == 2  # rel_pos_x, rel_pos_y
        assert (
            extractor._obstacle_size == 7
        )  # radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
        assert extractor.features_dim == 66  # target_size + d_model

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(15, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(15,)),
            }
        )

        extractor = AttentionExtractor(
            observation_space,
            d_model=128,
            num_heads=8,
            num_layers=2,
            max_obstacles=15,
            include_acceleration=False,
            include_radius=False,
        )

        assert extractor._d_model == 128
        assert extractor._num_heads == 8
        assert extractor._num_layers == 2
        assert extractor._max_obstacles == 15
        assert extractor._include_acceleration == False
        assert extractor._include_radius == False
        assert extractor._agent_size == 2  # vel_x, vel_y only
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4  # pos_x, pos_y, vel_x, vel_y only
        assert extractor.features_dim == 130  # target_size + d_model

    def test_layer_construction(self):
        """Test that all layers are properly constructed."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(10, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

        extractor = AttentionExtractor(observation_space, num_layers=3)

        assert len(extractor._q_projections) == 3
        assert len(extractor._k_projections) == 3
        assert len(extractor._v_projections) == 3
        assert len(extractor._mha_layers) == 3
        assert len(extractor._layer_norms) == 3
        assert len(extractor._feed_forwards) == 3

        # Check layer dimensions with default settings (include_acceleration=True, include_radius=True)
        assert (
            extractor._initial_agent_projection.in_features == 5
        )  # vel_x, vel_y, acc_x, acc_y, radius
        assert extractor._initial_agent_projection.out_features == 64
        assert extractor._output_projection.in_features == 64
        assert extractor._output_projection.out_features == 64


class TestAttentionExtractorForward:
    """Test suite for AttentionExtractor forward pass with batch dimensions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(10, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )
        self.extractor = AttentionExtractor(self.observation_space)
        self.batch_size = 4

    def create_test_observations(self, batch_size=None):
        """Create test observations with proper dimensions."""
        if batch_size is None:
            batch_size = self.batch_size

        observations = {
            "agent": torch.randn(batch_size, 5),
            "obstacles": torch.randn(batch_size, 10, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.randint(0, 2, (batch_size, 10)).float(),
        }
        return observations

    def test_forward_basic(self):
        """Test basic forward pass with default parameters."""
        observations = self.create_test_observations()

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        # Mock return values with correct dimensions for default settings
                        mock_agent.return_value = torch.randn(
                            self.batch_size,
                            5,  # vel_x, vel_y, acc_x, acc_y, radius
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 7  # radius, pos, vel, acc
                        )

                        output = self.extractor(observations)

                        assert output.shape == (
                            self.batch_size,
                            66,
                        )  # target_size + d_model
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16, 32]:
            observations = self.create_test_observations(batch_size)

            with patch(
                "extractors.attention_extractor.validate_observation_tensors"
            ):
                with patch(
                    "extractors.attention_extractor.extract_agent_features"
                ) as mock_agent:
                    with patch(
                        "extractors.attention_extractor.extract_target_features"
                    ) as mock_target:
                        with patch(
                            "extractors.attention_extractor.extract_obstacle_features"
                        ) as mock_obstacle:
                            mock_agent.return_value = torch.randn(batch_size, 5)
                            mock_target.return_value = torch.randn(
                                batch_size, 2
                            )
                            mock_obstacle.return_value = torch.randn(
                                batch_size, 10, 7
                            )

                            output = self.extractor(observations)

                            assert output.shape == (batch_size, 66)
                            assert not torch.isnan(output).any()
                            assert not torch.isinf(output).any()

    def test_forward_with_single_batch(self):
        """Test forward pass with single batch dimension."""
        observations = self.create_test_observations(batch_size=1)

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(1, 5)
                        mock_target.return_value = torch.randn(1, 2)
                        mock_obstacle.return_value = torch.randn(1, 10, 7)

                        output = self.extractor(observations)

                        assert output.shape == (1, 66)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_all_masked_obstacles(self):
        """Test forward pass when all obstacles are masked."""
        observations = self.create_test_observations()
        observations["mask"] = torch.zeros(self.batch_size, 10)  # All masked

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 5
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.zeros(
                            self.batch_size, 10, 7
                        )

                        output = self.extractor(observations)

                        assert output.shape == (self.batch_size, 66)
                        assert not torch.isnan(output).any()

    def test_forward_with_acceleration_and_no_radius(self):
        """Test forward pass with acceleration enabled and radius disabled."""
        extractor = AttentionExtractor(
            self.observation_space,
            include_acceleration=True,
            include_radius=False,
        )

        observations = self.create_test_observations()

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 4
                        )  # vel + acc (no radius)
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 6
                        )  # pos + vel + acc (no radius)

                        output = extractor(observations)

                        assert output.shape == (self.batch_size, 66)
                        assert not torch.isnan(output).any()

    def test_forward_with_no_acceleration_and_radius(self):
        """Test forward pass with acceleration disabled and radius enabled."""
        extractor = AttentionExtractor(
            self.observation_space,
            include_acceleration=False,
            include_radius=True,
        )

        observations = self.create_test_observations()

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 3
                        )  # vel + radius (no acc)
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 5
                        )  # radius + pos + vel (no acc)

                        output = extractor(observations)

                        assert output.shape == (self.batch_size, 66)
                        assert not torch.isnan(output).any()

    def test_forward_with_minimal_features(self):
        """Test forward pass with minimal features (no acceleration, no radius)."""
        extractor = AttentionExtractor(
            self.observation_space,
            include_acceleration=False,
            include_radius=False,
        )

        observations = self.create_test_observations()

        with patch(
            "extractors.attention_extractor.validate_observation_tensors"
        ):
            with patch(
                "extractors.attention_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.attention_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.attention_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 2
                        )  # vel only
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 4
                        )  # pos + vel only

                        output = extractor(observations)

                        assert output.shape == (self.batch_size, 66)
                        assert not torch.isnan(output).any()


class TestAttentionMechanism:
    """Test suite for attention mechanism specific functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(5, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(5,)),
            }
        )
        self.extractor = AttentionExtractor(observation_space, max_obstacles=5)

    def test_create_attention_mask(self):
        """Test attention mask creation."""
        mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0]])
        attention_mask = self.extractor._create_attention_mask(mask)

        expected = torch.tensor([[False, False, True, False, True]])
        torch.testing.assert_close(attention_mask, expected)

    def test_apply_attention_shape(self):
        """Test attention application output shape."""
        batch_size = 2
        agent_features = torch.randn(batch_size, 3)
        obstacle_features = torch.randn(batch_size, 5, 5)
        mask = torch.ones(batch_size, 5)

        output = self.extractor._apply_attention(
            agent_features, obstacle_features, mask
        )

        assert output.shape == (batch_size, 64)

    def test_apply_attention_with_masked_obstacles(self):
        """Test attention with some obstacles masked."""
        batch_size = 2
        agent_features = torch.randn(batch_size, 3)
        obstacle_features = torch.randn(batch_size, 5, 5)
        mask = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]]
        )

        output = self.extractor._apply_attention(
            agent_features, obstacle_features, mask
        )

        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()


class TestAttentionExtractorIntegration:
    """Integration tests for AttentionExtractor."""

    def test_end_to_end_processing(self):
        """Test complete end-to-end processing."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(3, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(3,)),
            }
        )

        extractor = AttentionExtractor(
            observation_space, max_obstacles=3, d_model=32, num_heads=2
        )

        # Real observations
        observations = {
            "agent": torch.tensor(
                [[0.5, 0.1, 0.2, 0.01, 0.02]]
            ),  # [radius, vel_x, vel_y, acc_x, acc_y]
            "obstacles": torch.tensor(
                [
                    [
                        [
                            0.3,
                            1.0,
                            1.0,
                            -0.5,
                            -0.5,
                            0.1,
                            0.1,
                        ],  # Valid obstacle
                        [
                            0.4,
                            -1.0,
                            0.5,
                            0.2,
                            -0.1,
                            -0.05,
                            0.05,
                        ],  # Valid obstacle
                        [
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                            0.0,
                        ],  # Masked obstacle
                    ]
                ]
            ),
            "target": torch.tensor([[0.8, 0.6]]),  # Target relative position
            "mask": torch.tensor(
                [[1.0, 1.0, 0.0]]
            ),  # First two obstacles valid
        }

        output = extractor(observations)

        assert output.shape == (1, 34)  # target_size(2) + d_model(32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_gradient_flow(self):
        """Test that gradients flow properly through the extractor."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(3, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(3,)),
            }
        )

        extractor = AttentionExtractor(observation_space, max_obstacles=3)

        observations = {
            "agent": torch.tensor(
                [[0.5, 0.1, 0.2, 0.01, 0.02]], requires_grad=True
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [
                            [0.3, 1.0, 1.0, -0.5, -0.5, 0.1, 0.1],
                            [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
                requires_grad=True,
            ),
            "target": torch.tensor([[0.8, 0.6]], requires_grad=True),
            "mask": torch.tensor([[1.0, 1.0, 0.0]]),
        }

        output = extractor(observations)
        loss = output.sum()
        loss.backward()

        # Check that gradients are computed
        assert observations["agent"].grad is not None
        assert observations["obstacles"].grad is not None
        assert observations["target"].grad is not None

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(2, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(2,)),
            }
        )

        extractor = AttentionExtractor(observation_space, max_obstacles=2)

        observations = {
            "agent": torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]]),
            "obstacles": torch.tensor(
                [
                    [
                        [
                            [0.3, 1.0, 1.0, -0.5, -0.5, 0.1, 0.1],
                            [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                        ]
                    ]
                ]
            ),
            "target": torch.tensor([[0.8, 0.6]]),
            "mask": torch.tensor([[1.0, 1.0]]),
        }

        extractor.eval()
        output1 = extractor(observations)
        output2 = extractor(observations)

        torch.testing.assert_close(output1, output2)
