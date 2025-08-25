"""
Comprehensive tests for LSTMExtractor.

This module tests the LSTMExtractor class including initialization,
forward pass, LSTM sequence processing, bidirectional processing,
and gradient flow.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch

from ..lstm_extractor import LSTMExtractor


class TestLSTMExtractor:
    """Test suite for LSTMExtractor class."""

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
        extractor = LSTMExtractor(observation_space)

        assert extractor._max_obstacles == 10
        assert extractor._include_acceleration == False
        assert extractor._bidirectional == False
        assert extractor._features_dim == 64  # Default features_dim
        assert isinstance(extractor._lstm, nn.LSTM)
        assert isinstance(extractor._post, nn.Sequential)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=5,
            include_acceleration=True,
            lstm_hidden=256,
            lstm_layers=2,
            bidirectional=True,
            features_dim=128,
        )

        assert extractor._max_obstacles == 5
        assert extractor._include_acceleration == True
        assert extractor._bidirectional == True
        assert extractor._features_dim == 128
        assert extractor._lstm.hidden_size == 256
        assert extractor._lstm.num_layers == 2
        assert extractor._lstm.bidirectional == True

    def test_initialization_with_layernorm(self):
        """Test initialization with layer normalization."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, use_layernorm=True)

        # Check that post-processing includes LayerNorm
        has_layernorm = any(
            isinstance(module, nn.LayerNorm)
            for module in extractor._post.modules()
        )
        assert has_layernorm

    def test_initialization_without_layernorm(self):
        """Test initialization without layer normalization."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, use_layernorm=False)

        # Check that post-processing doesn't include LayerNorm
        has_layernorm = any(
            isinstance(module, nn.LayerNorm)
            for module in extractor._post.modules()
        )
        assert not has_layernorm

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(obs)

        assert result.shape == (2, 64)  # batch_size, features_dim
        assert torch.isfinite(result).all()

    def test_forward_unidirectional_lstm(self):
        """Test forward pass with unidirectional LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            bidirectional=False,
            features_dim=64,
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_forward_bidirectional_lstm(self):
        """Test forward pass with bidirectional LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            bidirectional=True,
            features_dim=64,
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            include_acceleration=True,
            features_dim=64,
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_forward_with_multi_layer_lstm(self):
        """Test forward pass with multi-layer LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_layers=3,
            features_dim=64,
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_forward_with_masked_obstacles(self):
        """Test forward pass with partially masked obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        # Set some obstacles as invalid
        obs["mask"] = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_forward_all_obstacles_masked(self):
        """Test forward pass with all obstacles masked."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        # Set all obstacles as invalid
        obs["mask"] = torch.zeros(2, 3)

        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_gradient_flow(self):
        """Test gradient flow through the network."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32, features_dim=64
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)
        # Set all obstacles as valid for consistent gradient flow
        obs["mask"] = torch.ones(2, 3)

        for k in ("agent", "obstacles", "target"):
            obs[k].requires_grad_(True)

        out = extractor.forward(obs)
        loss = out.sum()
        loss.backward()

        # Check input gradients
        for k in ("agent", "obstacles", "target"):
            assert obs[k].grad is not None, f"{k} gradient is None"
            assert torch.any(obs[k].grad != 0), f"{k} gradient is all zeros"

        # Check parameter gradients
        param_count = 0
        grad_count = 0
        for name, param in extractor.named_parameters():
            if param.requires_grad:
                param_count += 1
                if param.grad is not None and torch.any(param.grad != 0):
                    grad_count += 1

        assert grad_count > 0, "No parameters received gradients"
        assert (
            grad_count >= param_count * 0.8
        ), f"Only {grad_count}/{param_count} parameters have gradients"

    def test_deterministic_output(self):
        """Test that output is deterministic given the same input."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )

        obs = self.create_test_observations(batch_size=2, max_obstacles=3)

        # Set deterministic seed
        torch.manual_seed(42)
        result1 = extractor.forward(obs)

        torch.manual_seed(42)
        result2 = extractor.forward(obs)

        assert torch.allclose(result1, result2, atol=1e-6)

    def test_batch_independence(self):
        """Test that different batch items are processed independently."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )

        # Create observations where second batch item is copy of first
        obs1 = self.create_test_observations(batch_size=1, max_obstacles=3)
        obs2 = {k: torch.cat([v, v], dim=0) for k, v in obs1.items()}

        result1 = extractor.forward(obs1)
        result2 = extractor.forward(obs2)

        # First batch result should be same
        assert torch.allclose(result1[0], result2[0], atol=1e-6)
        # Second batch result should be same as first (since they're identical)
        assert torch.allclose(result2[0], result2[1], atol=1e-6)

    def test_different_lstm_configurations(self):
        """Test different LSTM configurations produce different outputs."""
        observation_space = self.create_test_observation_space()
        obs = self.create_test_observations(batch_size=2, max_obstacles=3)

        # Unidirectional LSTM
        extractor_uni = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            bidirectional=False,
            features_dim=64,
        )

        # Bidirectional LSTM
        extractor_bi = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            bidirectional=True,
            features_dim=64,
        )

        result_uni = extractor_uni.forward(obs)
        result_bi = extractor_bi.forward(obs)

        # Results should be different due to different architectures
        assert not torch.allclose(result_uni, result_bi, atol=1e-3)
        assert result_uni.shape == result_bi.shape == (2, 64)

    def test_features_dim_property(self):
        """Test that features_dim property returns correct value."""
        observation_space = self.create_test_observation_space()

        extractor64 = LSTMExtractor(observation_space, features_dim=64)
        extractor128 = LSTMExtractor(observation_space, features_dim=128)

        assert extractor64.features_dim == 64
        assert extractor128.features_dim == 128

    def test_different_features_dim_outputs(self):
        """Test that different features_dim produces different output shapes."""
        observation_space = self.create_test_observation_space()
        obs = self.create_test_observations(batch_size=2, max_obstacles=3)

        extractor64 = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )
        extractor128 = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=128
        )

        result64 = extractor64.forward(obs)
        result128 = extractor128.forward(obs)

        assert result64.shape == (2, 64)
        assert result128.shape == (2, 128)

    @patch("extractors.lstm_extractor.validate_observation_tensors")
    def test_validation_called(self, mock_validate):
        """Test that observation validation is called."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=3)
        obs = self.create_test_observations(batch_size=2, max_obstacles=3)

        extractor.forward(obs)
        mock_validate.assert_called_once()


class TestLSTMExtractorIntegration:
    """Integration tests for LSTMExtractor with realistic scenarios."""

    def create_realistic_observations(self):
        """Create realistic observation data."""
        return {
            "agent": torch.tensor(
                [
                    [
                        0.5,
                        1.0,
                        2.0,
                        0.1,
                        -0.1,
                        0.01,
                        -0.01,
                    ],  # radius, pos, vel, acc
                    [0.5, -1.0, 1.5, -0.05, 0.2, 0.02, 0.0],
                ]
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],  # Static obstacle
                        [0.4, 0.5, 2.5, 0.1, 0.1, 0.0, 0.0],  # Moving obstacle
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Padding
                    ],
                    [
                        [0.2, -0.5, 2.0, 0.0, 0.0, 0.0, 0.0],  # Static obstacle
                        [0.3, 1.0, 1.0, -0.1, 0.0, 0.0, 0.0],  # Moving obstacle
                        [
                            0.6,
                            0.0,
                            3.0,
                            0.2,
                            -0.1,
                            0.0,
                            0.0,
                        ],  # Another moving obstacle
                    ],
                ]
            ),
            "target": torch.tensor([[3.0, 4.0], [2.0, 3.0]]),
            "mask": torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
        }

    def test_realistic_navigation_scenario(self):
        """Test with realistic navigation scenario data."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-float("inf"), float("inf"), (7,)),
                "obstacles": spaces.Box(-float("inf"), float("inf"), (3, 7)),
                "target": spaces.Box(-float("inf"), float("inf"), (2,)),
                "mask": spaces.Box(0, 1, (3,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, features_dim=64
        )
        obs = self.create_realistic_observations()

        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()

    def test_performance_large_batch(self):
        """Test performance with large batch size."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-float("inf"), float("inf"), (7,)),
                "obstacles": spaces.Box(-float("inf"), float("inf"), (5, 7)),
                "target": spaces.Box(-float("inf"), float("inf"), (2,)),
                "mask": spaces.Box(0, 1, (5,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=5, features_dim=64
        )

        batch_size = 32
        obs = {
            "agent": torch.randn(batch_size, 7),
            "obstacles": torch.randn(batch_size, 5, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.randint(0, 2, (batch_size, 5)).float(),
        }

        result = extractor.forward(obs)

        assert result.shape == (batch_size, 64)
        assert torch.isfinite(result).all()

    def test_edge_case_single_obstacle(self):
        """Test edge case with single obstacle."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(-float("inf"), float("inf"), (7,)),
                "obstacles": spaces.Box(-float("inf"), float("inf"), (1, 7)),
                "target": spaces.Box(-float("inf"), float("inf"), (2,)),
                "mask": spaces.Box(0, 1, (1,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=1, features_dim=64
        )

        obs = {
            "agent": torch.randn(2, 7),
            "obstacles": torch.randn(2, 1, 7),
            "target": torch.randn(2, 2),
            "mask": torch.ones(2, 1),
        }

        result = extractor.forward(obs)

        assert result.shape == (2, 64)
        assert torch.isfinite(result).all()


if __name__ == "__main__":
    pytest.main([__file__])
