"""Comprehensive tests for LSTMExtractor.

This module tests the LSTMExtractor functionality including initialization,
forward pass, LSTM processing, and feature extraction with proper batch dimensions.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch, MagicMock

from ..lstm_extractor import LSTMExtractor


class TestLSTMExtractorInit:
    """Test suite for LSTMExtractor initialization."""

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

        extractor = LSTMExtractor(observation_space)

        # Test default values
        assert extractor._max_obstacles == 10
        assert (
            extractor._include_acceleration == True
        )  # Corrected default value
        assert extractor._include_radius == True
        assert extractor._bidirectional == False
        assert extractor._agent_size == 5  # vel_x, vel_y, acc_x, acc_y, radius
        assert extractor._target_size == 2  # rel_pos_x, rel_pos_y
        assert (
            extractor._obstacle_size == 7
        )  # radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y
        assert extractor.features_dim == 64

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

        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=15,
            include_acceleration=False,
            include_radius=False,
            lstm_hidden=256,
            lstm_layers=2,
            bidirectional=True,
            use_layernorm=False,
            features_dim=128,
        )

        assert extractor._max_obstacles == 15
        assert extractor._include_acceleration == False
        assert extractor._include_radius == False
        assert extractor._bidirectional == True
        assert extractor._agent_size == 2  # vel_x, vel_y only
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4  # pos_x, pos_y, vel_x, vel_y only
        assert extractor.features_dim == 128

    def test_lstm_layer_construction(self):
        """Test that LSTM layer is properly constructed."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-1, high=1, shape=(5,)),
                "obstacles": spaces.Box(low=-1, high=1, shape=(10, 7)),
                "target": spaces.Box(low=-1, high=1, shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, lstm_hidden=64, lstm_layers=2, bidirectional=True
        )

        lstm = extractor._lstm
        assert (
            lstm.input_size == 7
        )  # obstacle feature size with default settings
        assert lstm.hidden_size == 64
        assert lstm.num_layers == 2
        assert lstm.batch_first == True
        assert lstm.bidirectional == True


class TestLSTMExtractorForward:
    """Test suite for LSTMExtractor forward pass with batch dimensions."""

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
        self.extractor = LSTMExtractor(self.observation_space)
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

        with patch("extractors.lstm_extractor.validate_observation_tensors"):
            with patch(
                "extractors.lstm_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.lstm_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.lstm_extractor.extract_obstacle_features"
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

                        assert output.shape == (self.batch_size, 64)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_different_batch_sizes(self):
        """Test forward pass with different batch sizes."""
        for batch_size in [1, 2, 8, 16, 32]:
            observations = self.create_test_observations(batch_size)

            with patch(
                "extractors.lstm_extractor.validate_observation_tensors"
            ):
                with patch(
                    "extractors.lstm_extractor.extract_agent_features"
                ) as mock_agent:
                    with patch(
                        "extractors.lstm_extractor.extract_target_features"
                    ) as mock_target:
                        with patch(
                            "extractors.lstm_extractor.extract_obstacle_features"
                        ) as mock_obstacle:
                            mock_agent.return_value = torch.randn(batch_size, 5)
                            mock_target.return_value = torch.randn(
                                batch_size, 2
                            )
                            mock_obstacle.return_value = torch.randn(
                                batch_size, 10, 7
                            )

                            output = self.extractor(observations)

                            assert output.shape == (batch_size, 64)
                            assert not torch.isnan(output).any()
                            assert not torch.isinf(output).any()

    def test_forward_with_bidirectional_lstm(self):
        """Test forward pass with bidirectional LSTM."""
        extractor = LSTMExtractor(
            self.observation_space, bidirectional=True, features_dim=128
        )

        observations = self.create_test_observations()

        with patch("extractors.lstm_extractor.validate_observation_tensors"):
            with patch(
                "extractors.lstm_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.lstm_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.lstm_extractor.extract_obstacle_features"
                    ) as mock_obstacle:
                        mock_agent.return_value = torch.randn(
                            self.batch_size, 5
                        )
                        mock_target.return_value = torch.randn(
                            self.batch_size, 2
                        )
                        mock_obstacle.return_value = torch.randn(
                            self.batch_size, 10, 7
                        )

                        output = extractor(observations)

                        assert output.shape == (self.batch_size, 128)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()

    def test_forward_with_minimal_features(self):
        """Test forward pass with minimal features (no acceleration, no radius)."""
        extractor = LSTMExtractor(
            self.observation_space,
            include_acceleration=False,
            include_radius=False,
        )

        observations = self.create_test_observations()

        with patch("extractors.lstm_extractor.validate_observation_tensors"):
            with patch(
                "extractors.lstm_extractor.extract_agent_features"
            ) as mock_agent:
                with patch(
                    "extractors.lstm_extractor.extract_target_features"
                ) as mock_target:
                    with patch(
                        "extractors.lstm_extractor.extract_obstacle_features"
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

                        assert output.shape == (self.batch_size, 64)
                        assert not torch.isnan(output).any()
                        assert not torch.isinf(output).any()
