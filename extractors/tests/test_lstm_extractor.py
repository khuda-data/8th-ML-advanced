"""
Comprehensive tests for LSTMExtractor.

This module tests the LSTMExtractor class including initialization,
forward pass, LSTM sequence processing, and packed sequence handling.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch, MagicMock

from ..lstm_extractor import LSTMExtractor


class TestLSTMExtractor:
    """Test suite for LSTMExtractor class."""

    def create_test_observation_space(self):
        """Create a test observation space for the extractor."""
        return spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(10, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

    def create_test_observations(
        self, batch_size: int = 2, max_obstacles: int = 10
    ):
        """Create test observation tensors."""
        return {
            "agent": torch.randn(batch_size, 7),
            "obstacles": torch.randn(batch_size, max_obstacles, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.randint(0, 2, (batch_size, max_obstacles)).float(),
        }

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        observation_space = self.create_test_observation_space()

        extractor = LSTMExtractor(observation_space)

        assert extractor._max_obstacles == 10
        assert extractor._include_acceleration == False
        assert extractor._agent_size == 2
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4
        assert extractor._obstacles_total_size == 40

        # Check LSTM configuration
        assert isinstance(extractor._lstm, nn.LSTM)
        assert extractor._lstm.input_size == 4
        assert extractor._lstm.hidden_size == 128
        assert extractor._lstm.num_layers == 1
        assert extractor._lstm.batch_first == True
        assert extractor._lstm.bidirectional == False

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
            use_layernorm=False,
        )

        assert extractor._max_obstacles == 5
        assert extractor._include_acceleration == True
        assert extractor._agent_size == 4  # with acceleration
        assert extractor._obstacle_size == 6  # with acceleration

        # Check LSTM configuration
        assert extractor._lstm.input_size == 6
        assert extractor._lstm.hidden_size == 256
        assert extractor._lstm.num_layers == 2
        assert extractor._lstm.bidirectional == True

        # Check output projection
        lstm_out_dim = 256 * 2  # bidirectional
        assert extractor._per_step_head.in_features == lstm_out_dim
        assert extractor._per_step_head.out_features == 6

        # Check post-processing layer
        assert isinstance(extractor._post, nn.Identity)

    def test_initialization_with_layernorm(self):
        """Test initialization with layer normalization."""
        observation_space = self.create_test_observation_space()

        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, use_layernorm=True
        )

        expected_dim = 2 + 2 + (4 * 3)  # agent + target + obstacles
        assert isinstance(extractor._post, nn.LayerNorm)
        assert extractor._post.normalized_shape == (expected_dim,)

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result = extractor.forward(observations)

        expected_shape = (2, extractor._features_dim)
        assert result.shape == expected_shape
        assert torch.all(torch.isfinite(result))
        assert not torch.all(result == 0)

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            include_acceleration=True,
            lstm_hidden=32,
        )

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        result = extractor.forward(observations)

        # Expected dimensions: agent(4) + target(2) + obstacles(6*3) = 24
        expected_shape = (1, 24)
        assert result.shape == expected_shape

    def test_forward_with_bidirectional_lstm(self):
        """Test forward pass with bidirectional LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            bidirectional=True,
        )

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result = extractor.forward(observations)

        assert result.shape == (2, extractor._features_dim)
        assert torch.all(torch.isfinite(result))

    def test_forward_with_multi_layer_lstm(self):
        """Test forward pass with multi-layer LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32, lstm_layers=3
        )

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result = extractor.forward(observations)

        assert result.shape == (2, extractor._features_dim)
        assert torch.all(torch.isfinite(result))

    def test_forward_sequence_packing(self):
        """Test that LSTM handles variable sequence lengths through packing."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=4, lstm_hidden=32
        )

        # Create observations with different valid sequence lengths
        observations = {
            "agent": torch.randn(3, 7),
            "obstacles": torch.randn(3, 4, 7),
            "target": torch.randn(3, 2),
            "mask": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],  # 4 valid obstacles
                    [1.0, 1.0, 0.0, 0.0],  # 2 valid obstacles
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            ),  # 1 valid obstacle
        }

        result = extractor.forward(observations)

        assert result.shape == (3, extractor._features_dim)
        assert torch.all(torch.isfinite(result))

    def test_forward_with_masked_obstacles(self):
        """Test forward pass with some obstacles masked out."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=4, lstm_hidden=32
        )

        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 4, 7),
            "target": torch.randn(1, 2),
            "mask": torch.tensor(
                [[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32
            ),  # Last 2 obstacles masked
        }

        result = extractor.forward(observations)

        # Extract obstacle portion and check that masked obstacles have zero features
        obstacle_start = extractor._agent_size + extractor._target_size
        obstacle_features = result[:, obstacle_start:].reshape(1, 4, 4)

        # Last 2 obstacles should have zero features
        torch.testing.assert_close(
            obstacle_features[:, 2:, :], torch.zeros(1, 2, 4)
        )

    def test_forward_all_obstacles_masked(self):
        """Test behavior when all obstacles are masked."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 3, 7),
            "target": torch.randn(1, 2),
            "mask": torch.zeros(1, 3),  # All obstacles masked
        }

        result = extractor.forward(observations)

        # Should still produce valid output
        assert result.shape == (1, extractor._features_dim)
        assert torch.all(torch.isfinite(result))

        # Obstacle portion should be zeros
        obstacle_start = extractor._agent_size + extractor._target_size
        obstacle_features = result[:, obstacle_start:]
        torch.testing.assert_close(
            obstacle_features, torch.zeros(1, extractor._obstacles_total_size)
        )

    def test_lstm_output_projection(self):
        """Test that LSTM output is properly projected back to obstacle feature size."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=64,
            bidirectional=False,
        )

        # Check projection layer
        assert extractor._per_step_head.in_features == 64
        assert (
            extractor._per_step_head.out_features == 4
        )  # obstacle_size without acceleration

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )
        result = extractor.forward(observations)

        # Check that obstacle features have correct dimensions after projection
        obstacle_start = extractor._agent_size + extractor._target_size
        obstacle_features = result[:, obstacle_start:].reshape(1, 3, 4)
        assert obstacle_features.shape == (1, 3, 4)

    def test_gradient_flow(self):
        """Test that gradients can flow through the LSTM."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        # Enable gradient computation
        for key in observations:
            observations[key].requires_grad_(True)

        result = extractor.forward(observations)
        loss = result.sum()
        loss.backward()

        # Check that gradients exist
        for key in observations:
            assert observations[key].grad is not None
            assert torch.any(observations[key].grad != 0)

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        observations = self.create_test_observations(
            batch_size=2, max_obstacles=3
        )

        result1 = extractor.forward(observations)
        result2 = extractor.forward(observations)

        torch.testing.assert_close(result1, result2)

    def test_batch_independence(self):
        """Test that different batch items are processed independently."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        observations_batch = self.create_test_observations(
            batch_size=3, max_obstacles=3
        )

        result_batch = extractor.forward(observations_batch)

        # Process each item individually
        results_individual = []
        for i in range(3):
            obs_single = {
                key: val[i : i + 1] for key, val in observations_batch.items()
            }
            result_single = extractor.forward(obs_single)
            results_individual.append(result_single)

        result_individual_stacked = torch.cat(results_individual, dim=0)

        # Should be identical
        torch.testing.assert_close(
            result_batch, result_individual_stacked, rtol=1e-5, atol=1e-6
        )

    def test_lstm_hidden_state_initialization(self):
        """Test that LSTM hidden states are properly initialized."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32, lstm_layers=2
        )

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        # Multiple forward passes should be independent (no hidden state carryover)
        result1 = extractor.forward(observations)
        result2 = extractor.forward(observations)

        torch.testing.assert_close(result1, result2)

    def test_different_lstm_configurations(self):
        """Test different LSTM configuration combinations."""
        observation_space = self.create_test_observation_space()

        configs = [
            {"lstm_hidden": 32, "lstm_layers": 1, "bidirectional": False},
            {"lstm_hidden": 64, "lstm_layers": 2, "bidirectional": False},
            {"lstm_hidden": 32, "lstm_layers": 1, "bidirectional": True},
            {"lstm_hidden": 64, "lstm_layers": 2, "bidirectional": True},
        ]

        for config in configs:
            extractor = LSTMExtractor(
                observation_space, max_obstacles=3, **config
            )

            observations = self.create_test_observations(
                batch_size=1, max_obstacles=3
            )
            result = extractor.forward(observations)

            assert result.shape == (1, extractor._features_dim)
            assert torch.all(torch.isfinite(result))

    @patch("extractors.lstm_extractor.validate_observation_tensors")
    def test_forward_calls_validation(self, mock_validate):
        """Test that forward pass calls validation function."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        extractor.forward(observations)

        # Check that validation was called
        mock_validate.assert_called_once()

    def test_features_dim_property(self):
        """Test that features_dim property returns correct value."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=5, include_acceleration=True
        )

        expected_dim = 4 + 2 + (6 * 5)  # agent(vel+acc) + target + obstacles
        assert extractor.features_dim == expected_dim
        assert extractor._features_dim == expected_dim

    def test_sequence_length_handling(self):
        """Test proper handling of sequence lengths."""
        observation_space = self.create_test_observation_space()
        extractor = LSTMExtractor(
            observation_space, max_obstacles=5, lstm_hidden=32
        )

        # Test with various sequence lengths
        masks = [
            torch.tensor(
                [[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float32
            ),  # All valid
            torch.tensor(
                [[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32
            ),  # 3 valid
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),  # 1 valid
        ]

        for mask in masks:
            observations = {
                "agent": torch.randn(1, 7),
                "obstacles": torch.randn(1, 5, 7),
                "target": torch.randn(1, 2),
                "mask": mask,
            }

            result = extractor.forward(observations)

            assert result.shape == (1, extractor._features_dim)
            assert torch.all(torch.isfinite(result))

    def test_layer_normalization_effect(self):
        """Test the effect of layer normalization."""
        observation_space = self.create_test_observation_space()

        extractor_with_ln = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            use_layernorm=True,
        )

        extractor_without_ln = LSTMExtractor(
            observation_space,
            max_obstacles=3,
            lstm_hidden=32,
            use_layernorm=False,
        )

        observations = self.create_test_observations(
            batch_size=1, max_obstacles=3
        )

        result_with_ln = extractor_with_ln.forward(observations)
        result_without_ln = extractor_without_ln.forward(observations)

        # Both should produce valid outputs
        assert torch.all(torch.isfinite(result_with_ln))
        assert torch.all(torch.isfinite(result_without_ln))

        # Results should be different (unless very unlikely coincidence)
        assert not torch.allclose(result_with_ln, result_without_ln, atol=1e-3)


class TestLSTMExtractorIntegration:
    """Integration tests for LSTMExtractor with realistic scenarios."""

    def create_realistic_observations(self):
        """Create realistic navigation scenario observations."""
        return {
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

    def test_realistic_navigation_scenario(self):
        """Test with realistic navigation scenario."""
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

        extractor = LSTMExtractor(
            observation_space, max_obstacles=5, lstm_hidden=64, lstm_layers=2
        )

        observations = self.create_realistic_observations()

        result = extractor.forward(observations)

        expected_dim = 2 + 2 + (4 * 5)  # agent + target + obstacles
        assert result.shape == (1, expected_dim)
        assert torch.all(torch.isfinite(result))
        assert not torch.all(result == 0)

    def test_performance_large_batch(self):
        """Test performance with large batch size."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(10, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=10, lstm_hidden=128
        )

        batch_size = 64
        observations = {
            "agent": torch.randn(batch_size, 7),
            "obstacles": torch.randn(batch_size, 10, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, 10),
        }

        result = extractor.forward(observations)

        expected_dim = 2 + 2 + (4 * 10)
        assert result.shape == (batch_size, expected_dim)
        assert torch.all(torch.isfinite(result))

    def test_lstm_sequence_order_sensitivity(self):
        """Test that LSTM is sensitive to obstacle sequence order."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(3, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(3,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=3, lstm_hidden=32
        )

        # Original sequence
        observations1 = {
            "agent": torch.tensor(
                [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.3, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "target": torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            "mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        }

        # Reordered sequence
        observations2 = {
            "agent": observations1["agent"].clone(),
            "obstacles": torch.tensor(
                [
                    [
                        [0.3, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
            "target": observations1["target"].clone(),
            "mask": observations1["mask"].clone(),
        }

        result1 = extractor.forward(observations1)
        result2 = extractor.forward(observations2)

        # LSTM should be sensitive to sequence order
        assert not torch.allclose(result1, result2, atol=1e-3)

    def test_edge_case_single_obstacle(self):
        """Test edge case with only one obstacle."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(7,)
                ),
                "obstacles": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(1, 7)
                ),
                "target": spaces.Box(
                    low=-float("inf"), high=float("inf"), shape=(2,)
                ),
                "mask": spaces.Box(low=0, high=1, shape=(1,)),
            }
        )

        extractor = LSTMExtractor(
            observation_space, max_obstacles=1, lstm_hidden=32
        )

        observations = {
            "agent": torch.randn(2, 7),
            "obstacles": torch.randn(2, 1, 7),
            "target": torch.randn(2, 2),
            "mask": torch.ones(2, 1),
        }

        result = extractor.forward(observations)

        expected_dim = 2 + 2 + (4 * 1)
        assert result.shape == (2, expected_dim)
        assert torch.all(torch.isfinite(result))


if __name__ == "__main__":
    pytest.main([__file__])
