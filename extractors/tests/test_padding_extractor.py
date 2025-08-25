"""Comprehensive tests for PaddingExtractor.

This module tests the PaddingExtractor functionality including initialization,
forward pass, feature flattening, and feature extraction with proper batch dimensions.
"""

import pytest
import torch
from gymnasium import spaces
from unittest.mock import patch, MagicMock

from ..padding_extractor import PaddingExtractor


class TestPaddingExtractorInit:
    """Test suite for PaddingExtractor initialization."""

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

        extractor = PaddingExtractor(observation_space)

        # Test default values
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
        assert extractor._obstacles_total_size == 70  # 7 * 10
        assert extractor._features_dim == 77  # 5 + 2 + 70
        assert extractor.features_dim == 77

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

        extractor = PaddingExtractor(
            observation_space,
            max_obstacles=15,
            include_acceleration=False,
            include_radius=False,
        )

        assert extractor._max_obstacles == 15
        assert extractor._include_acceleration == False
        assert extractor._include_radius == False
        assert extractor._agent_size == 2  # vel_x, vel_y only
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4  # pos_x, pos_y, vel_x, vel_y only
        assert extractor._obstacles_total_size == 60  # 4 * 15
        assert extractor._features_dim == 64  # 2 + 2 + 60

    def test_feature_size_calculations(self):
        """Test feature size calculations for different configurations."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(5, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(5,)),
            }
        )

        # Test all features enabled (default)
        extractor_all = PaddingExtractor(
            observation_space,
            max_obstacles=5,
            include_acceleration=True,
            include_radius=True,
        )
        assert (
            extractor_all._agent_size == 5
        )  # vel_x, vel_y, acc_x, acc_y, radius
        assert (
            extractor_all._obstacle_size == 7
        )  # radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
        assert extractor_all._features_dim == 42  # 5 + 2 + 35

        # Test minimal features
        extractor_min = PaddingExtractor(
            observation_space,
            max_obstacles=5,
            include_acceleration=False,
            include_radius=False,
        )
        assert extractor_min._agent_size == 2  # vel_x, vel_y only
        assert (
            extractor_min._obstacle_size == 4
        )  # pos_x, pos_y, vel_x, vel_y only
        assert extractor_min._features_dim == 24  # 2 + 2 + 20

    def test_kwargs_parsing(self):
        """Test that kwargs are properly parsed."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(8, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(8,)),
            }
        )

        extractor = PaddingExtractor(
            observation_space,
            max_obstacles=8,
            include_acceleration=True,
            include_radius=False,
            some_unused_kwarg=42,  # Should be ignored
        )

        assert extractor._max_obstacles == 8
        assert extractor._include_acceleration == True
        assert extractor._include_radius == False


class TestPaddingExtractorForward:
    """Test suite for PaddingExtractor forward pass with batch dimensions."""

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
        self.extractor = PaddingExtractor(self.observation_space)
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

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
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
                            77,
                        )  # 5 + 2 + 70
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16, 32]:
            observations = self.create_test_observations(batch_size)

            with patch(
                "extractors.padding_extractor.validate_observation_tensors"
            ):
                with patch(
                    "extractors.padding_extractor.extract_agent_features"
                ) as mock_agent:
                    with patch(
                        "extractors.padding_extractor.extract_target_features"
                    ) as mock_target:
                        with patch(
                            "extractors.padding_extractor.extract_obstacle_features"
                        ) as mock_obstacle:
                            mock_agent.return_value = torch.randn(batch_size, 5)
                            mock_target.return_value = torch.randn(
                                batch_size, 2
                            )
                            mock_obstacle.return_value = torch.randn(
                                batch_size, 10, 7
                            )

                            output = self.extractor(observations)

                            assert output.shape == (batch_size, 77)
                            assert not torch.isnan(output).any()
                            assert not torch.isinf(output).any()

    def test_forward_with_single_batch(self):
        """Test forward pass with single batch dimension."""
        observations = self.create_test_observations(batch_size=1)

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(1, 5)
                        mock_target.return_value = torch.randn(1, 2)
                        mock_obstacle.return_value = torch.randn(1, 10, 7)

                        output = self.extractor(observations)

                        assert output.shape == (1, 77)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_acceleration_and_no_radius(self):
        """Test forward pass with acceleration enabled and radius disabled."""
        extractor = PaddingExtractor(
            self.observation_space,
            include_acceleration=True,
            include_radius=False,
        )

        observations = self.create_test_observations()

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 4  # vel + acc (no radius)
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size,
                            10,
                            6,  # pos + vel + acc (no radius)
                        )

                        output = extractor(observations)

                        expected_dim = 66  # 4 + 2 + (6 * 10) = 66
                        assert output.shape == (self.batch_size, expected_dim)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_no_acceleration_and_radius(self):
        """Test forward pass with acceleration disabled and radius enabled."""
        extractor = PaddingExtractor(
            self.observation_space,
            include_acceleration=False,
            include_radius=True,
        )

        observations = self.create_test_observations()

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 3  # vel + radius (no acc)
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size,
                            10,
                            5,  # radius + pos + vel (no acc)
                        )

                        output = extractor(observations)

                        expected_dim = 55  # 3 + 2 + (5 * 10)
                        assert output.shape == (self.batch_size, expected_dim)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_minimal_features(self):
        """Test forward pass with minimal features (no acceleration, no radius)."""
        extractor = PaddingExtractor(
            self.observation_space,
            include_acceleration=False,
            include_radius=False,
        )

        observations = self.create_test_observations()

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 2  # vel only
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 4  # pos + vel only
                        )

                        output = extractor(observations)

                        expected_dim = 44  # 2 + 2 + (4 * 10)
                        assert output.shape == (self.batch_size, expected_dim)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_all_masked_obstacles(self):
        """Test forward pass when all obstacles are masked."""
        observations = self.create_test_observations()
        observations["mask"] = torch.zeros(self.batch_size, 10)  # All masked

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
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

                        assert output.shape == (self.batch_size, 77)
                        assert not torch.isnan(output).any()

    def test_forward_custom_max_obstacles(self):
        """Test forward pass with custom max obstacles."""
        custom_observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(5, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(5,)),
            }
        )

        extractor = PaddingExtractor(custom_observation_space, max_obstacles=5)

        observations = {
            "agent": torch.randn(self.batch_size, 5),
            "obstacles": torch.randn(self.batch_size, 5, 7),
            "target": torch.randn(self.batch_size, 2),
            "mask": torch.randint(0, 2, (self.batch_size, 5)).float(),
        }

        with patch("extractors.padding_extractor.validate_observation_tensors"):
            with patch(
                "extractors.padding_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.padding_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.padding_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size,
                            5,  # radius + vel_x + vel_y + acc_x + acc_y (default extractor config)
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size,
                            5,
                            7,  # radius + rel_pos_x + rel_pos_y + rel_vel_x + rel_vel_y + acc_x + acc_y (default extractor config)
                        )

                        output = extractor(observations)

                        expected_dim = (
                            5 + 2 + (5 * 7)
                        )  # agent + target + obstacles = 5 + 2 + 35 = 42
                        assert output.shape == (self.batch_size, expected_dim)


class TestObstacleFlattening:
    """Test suite for obstacle feature flattening."""

    def setup_method(self):
        """Set up test fixtures."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(3, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(3,)),
            }
        )
        self.extractor = PaddingExtractor(observation_space, max_obstacles=3)

    def test_flatten_obstacle_features_basic(self):
        """Test basic obstacle feature flattening."""
        batch_size = 2
        max_obstacles = 3
        feature_size = 4

        obstacle_features = torch.tensor(
            [
                [  # Batch 1
                    [1.0, 2.0, 3.0, 4.0],  # Obstacle 1
                    [5.0, 6.0, 7.0, 8.0],  # Obstacle 2
                    [9.0, 10.0, 11.0, 12.0],  # Obstacle 3
                ],
                [  # Batch 2
                    [13.0, 14.0, 15.0, 16.0],  # Obstacle 1
                    [17.0, 18.0, 19.0, 20.0],  # Obstacle 2
                    [21.0, 22.0, 23.0, 24.0],  # Obstacle 3
                ],
            ]
        )

        flattened = self.extractor._flatten_obstacle_features(
            obstacle_features, max_obstacles, feature_size
        )

        assert flattened.shape == (batch_size, max_obstacles * feature_size)

        # Check flattening is correct
        expected_batch1 = torch.tensor(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        )
        expected_batch2 = torch.tensor(
            [
                13.0,
                14.0,
                15.0,
                16.0,
                17.0,
                18.0,
                19.0,
                20.0,
                21.0,
                22.0,
                23.0,
                24.0,
            ]
        )

        torch.testing.assert_close(flattened[0], expected_batch1)
        torch.testing.assert_close(flattened[1], expected_batch2)

    def test_flatten_obstacle_features_single_batch(self):
        """Test obstacle feature flattening with single batch."""
        obstacle_features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])

        flattened = self.extractor._flatten_obstacle_features(
            obstacle_features, max_obstacles=3, feature_size=2
        )

        assert flattened.shape == (1, 6)
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        torch.testing.assert_close(flattened, expected)

    def test_flatten_obstacle_features_zeros(self):
        """Test obstacle feature flattening with zero features."""
        batch_size = 2
        obstacle_features = torch.zeros(batch_size, 4, 3)

        flattened = self.extractor._flatten_obstacle_features(
            obstacle_features, max_obstacles=4, feature_size=3
        )

        assert flattened.shape == (batch_size, 12)
        assert torch.all(flattened == 0)

    def test_flatten_obstacle_features_different_shapes(self):
        """Test obstacle feature flattening with different input shapes."""
        # Test different combinations of max_obstacles and feature_size
        test_cases = [
            (1, 5, 2),  # 1 batch, 5 obstacles, 2 features each
            (3, 2, 7),  # 3 batch, 2 obstacles, 7 features each
            (4, 8, 1),  # 4 batch, 8 obstacles, 1 feature each
        ]

        for batch_size, max_obstacles, feature_size in test_cases:
            obstacle_features = torch.randn(
                batch_size, max_obstacles, feature_size
            )

            flattened = self.extractor._flatten_obstacle_features(
                obstacle_features, max_obstacles, feature_size
            )

            expected_shape = (batch_size, max_obstacles * feature_size)
            assert flattened.shape == expected_shape


class TestPaddingExtractorIntegration:
    """Integration tests for PaddingExtractor."""

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

        extractor = PaddingExtractor(observation_space, max_obstacles=3)

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

        expected_dim = (
            5 + 2 + (3 * 7)
        )  # agent + target + obstacles = 5 + 2 + 21 = 28
        assert output.shape == (1, expected_dim)
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

        extractor = PaddingExtractor(observation_space, max_obstacles=3)

        observations = {
            "agent": torch.tensor(
                [[0.5, 0.1, 0.2, 0.01, 0.02]], requires_grad=True
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 1.0, -0.5, -0.5, 0.1, 0.1],
                        [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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

        extractor = PaddingExtractor(observation_space, max_obstacles=2)

        observations = {
            "agent": torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]]),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 1.0, -0.5, -0.5, 0.1, 0.1],
                        [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                    ]
                ]
            ),
            "target": torch.tensor([[0.8, 0.6]]),
            "mask": torch.tensor([[1.0, 1.0]]),
        }

        output1 = extractor(observations)
        output2 = extractor(observations)

        torch.testing.assert_close(output1, output2)

    def test_feature_concatenation_order(self):
        """Test that features are concatenated in the correct order."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(2, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(2,)),
            }
        )

        extractor = PaddingExtractor(observation_space, max_obstacles=2)

        # Use specific values to check order
        observations = {
            "agent": torch.tensor([[0.5, 0.1, 0.2, 0.01, 0.02]]),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 1.0, -0.5, -0.5, 0.1, 0.1],
                        [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                    ]
                ]
            ),
            "target": torch.tensor([[0.8, 0.6]]),
            "mask": torch.tensor([[1.0, 1.0]]),
        }

        output = extractor(observations)

        # Expected structure: [agent_features, target_features, flattened_obstacles]
        # agent_features: [radius, vel_x, vel_y, acc_x, acc_y] = [0.5, 0.1, 0.2, 0.01, 0.02]
        # target_features: [0.8, 0.6]
        # obstacle_features: [radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y] for each obstacle

        # Check that agent features come first (all 5 features with default settings)
        torch.testing.assert_close(
            output[0, :5], torch.tensor([0.5, 0.1, 0.2, 0.01, 0.02])
        )
        # Check that target features come next
        torch.testing.assert_close(output[0, 5:7], torch.tensor([0.8, 0.6]))
        # Remaining should be flattened obstacle features
        assert output.shape[1] == 5 + 2 + (
            2 * 7
        )  # agent + target + obstacles = 5 + 2 + 14 = 21

    def test_different_configurations(self):
        """Test with different feature configurations."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(2, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(2,)),
            }
        )

        configurations = [
            {"include_acceleration": True, "include_radius": True},
            {"include_acceleration": True, "include_radius": False},
            {"include_acceleration": False, "include_radius": True},
            {"include_acceleration": False, "include_radius": False},
        ]

        base_observations = {
            "agent": torch.tensor(
                [[0.5, 0.1, 0.2, 0.01, 0.02]]
            ),  # radius, vel_x, vel_y, acc_x, acc_y
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
                        ],  # radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y
                        [0.4, -1.0, 0.5, 0.2, -0.1, -0.05, 0.05],
                    ]
                ]
            ),
            "target": torch.tensor([[0.8, 0.6]]),
            "mask": torch.tensor([[1.0, 1.0]]),
        }

        for config in configurations:
            extractor = PaddingExtractor(
                observation_space, max_obstacles=2, **config
            )

            # Create observation data that matches the configuration
            observations = {
                "target": base_observations["target"],
                "mask": base_observations["mask"],
            }

            # Adjust agent data based on configuration
            observations["agent"] = base_observations["agent"]  # All 5 features

            # Adjust obstacle data based on configuration
            observations["obstacles"] = base_observations[
                "obstacles"
            ]  # All 7 features

            output = extractor(observations)

            # Verify output shape matches expected feature dimensions
            assert output.shape == (1, extractor.features_dim)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
