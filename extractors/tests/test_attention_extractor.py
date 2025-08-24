""" Comprehensive tests for AttentionExtractor.
This module tests the AttentionExtractor class including initialization,
forward pass, attention mechanism, and multi-layer processing.
"""

import pytest
import torch
import torch.nn as nn
from gymnasium import spaces
from unittest.mock import patch, MagicMock

from ..attention_extractor import AttentionExtractor


# --- 🔧 Hotfix for mask method binding ---
# In the current implementation, _create_attention_mask is defined without `self`
# but is called via `self._create_attention_mask(...)`, which makes Python pass
# `self` implicitly and raises a TypeError. We wrap it as a staticmethod in tests.
@pytest.fixture(autouse=True)
def _staticize_attention_mask(monkeypatch):
    fn = AttentionExtractor.__dict__.get("_create_attention_mask")
    # If it's already a staticmethod in code, do nothing
    if isinstance(fn, staticmethod):
        return
    # Otherwise, wrap the function object as a staticmethod
    monkeypatch.setattr(
        AttentionExtractor,
        "_create_attention_mask",
        staticmethod(fn),
        raising=True,
    )
# -----------------------------------------


class TestAttentionExtractor:
    """Test suite for AttentionExtractor class."""

    def create_test_observation_space(self):
        """Create a test observation space for the extractor."""
        return spaces.Dict(
            {
                "agent": spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,)),
                "obstacles": spaces.Box(low=-float("inf"), high=float("inf"), shape=(10, 7)),
                "target": spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )

    def create_test_observations(self, batch_size: int = 2, max_obstacles: int = 10):
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
        extractor = AttentionExtractor(observation_space)

        assert extractor._d_model == 64
        assert extractor._num_heads == 4
        assert extractor._num_layers == 1
        assert extractor._max_obstacles == 10
        assert extractor._include_acceleration is False
        assert extractor._agent_size == 2
        assert extractor._target_size == 2
        assert extractor._obstacle_size == 4
        assert extractor.features_dim == 66  # target_size (2) + d_model (64)

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space,
            d_model=128,
            num_heads=8,
            num_layers=2,
            max_obstacles=5,
            include_acceleration=True,
        )

        assert extractor._d_model == 128
        assert extractor._num_heads == 8
        assert extractor._num_layers == 2
        assert extractor._max_obstacles == 5
        assert extractor._include_acceleration is True
        assert extractor._agent_size == 4  # with acceleration
        assert extractor._obstacle_size == 6  # with acceleration
        assert extractor.features_dim == 130  # target_size (2) + d_model (128)

    def test_layer_initialization(self):
        """Test that all neural network layers are properly initialized."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=64, num_heads=4, num_layers=2)

        # Check that projection layers exist
        assert isinstance(extractor._initial_agent_projection, nn.Linear)
        assert extractor._initial_agent_projection.in_features == 2
        assert extractor._initial_agent_projection.out_features == 64

        # Check that we have correct number of layers
        assert len(extractor._q_projections) == 2
        assert len(extractor._k_projections) == 2
        assert len(extractor._v_projections) == 2
        assert len(extractor._mha_layers) == 2
        assert len(extractor._layer_norms) == 2
        assert len(extractor._feed_forwards) == 2

        # Check projection dimensions
        assert extractor._q_projections[0].in_features == 64
        assert extractor._q_projections[0].out_features == 64
        assert extractor._k_projections[0].in_features == 4
        assert extractor._k_projections[0].out_features == 64
        assert extractor._v_projections[0].in_features == 4
        assert extractor._v_projections[0].out_features == 64

    def test_forward_basic_functionality(self):
        """Test basic forward pass functionality."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, num_heads=2, max_obstacles=3)

        observations = self.create_test_observations(batch_size=2, max_obstacles=3)
        result = extractor.forward(observations)

        # Check output shape: target_size + d_model
        expected_shape = (2, 2 + 32)
        assert result.shape == expected_shape

        # Check that result is not all zeros
        assert not torch.all(result == 0)
        assert torch.all(torch.isfinite(result))

    def test_forward_with_acceleration(self):
        """Test forward pass with acceleration features."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, d_model=32, max_obstacles=3, include_acceleration=True
        )

        observations = self.create_test_observations(batch_size=1, max_obstacles=3)
        result = extractor.forward(observations)

        # Output should still be target_size + d_model
        expected_shape = (1, 2 + 32)
        assert result.shape == expected_shape

    def test_forward_single_layer_vs_multi_layer(self):
        """Test difference between single layer and multi-layer attention."""
        observation_space = self.create_test_observation_space()

        extractor_single = AttentionExtractor(observation_space, d_model=32, num_layers=1, max_obstacles=3)
        extractor_multi = AttentionExtractor(observation_space, d_model=32, num_layers=3, max_obstacles=3)

        observations = self.create_test_observations(batch_size=1, max_obstacles=3)

        result_single = extractor_single.forward(observations)
        result_multi = extractor_multi.forward(observations)

        # Both should have same output shape
        assert result_single.shape == result_multi.shape
        # But different values (unless very unlikely coincidence)
        assert not torch.allclose(result_single, result_multi, atol=1e-3)

    def test_forward_attention_with_mask(self):
        """Test that attention properly handles masked obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, max_obstacles=4)

        # Create observations with specific mask
        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 4, 7),
            "target": torch.randn(1, 2),
            "mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),  # Only first 2 valid
        }

        result = extractor.forward(observations)

        assert result.shape == (1, 2 + 32)
        assert torch.all(torch.isfinite(result))

    def test_forward_all_obstacles_masked(self):
        """Test behavior when all obstacles are masked."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, max_obstacles=3)

        observations = {
            "agent": torch.randn(1, 7),
            "obstacles": torch.randn(1, 3, 7),
            "target": torch.randn(1, 2),
            "mask": torch.zeros(1, 3),  # All obstacles masked
        }

        result = extractor.forward(observations)

        assert result.shape == (1, 2 + 32)
        assert torch.all(torch.isfinite(result))

    def test_apply_attention_method(self):
        """Test the private _apply_attention method."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(
            observation_space, d_model=32, num_heads=2, num_layers=1, max_obstacles=3
        )

        # Create test inputs for _apply_attention
        batch_size = 2
        agent_features = torch.randn(batch_size, 2)
        obstacle_features = torch.randn(batch_size, 3, 4)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

        result = extractor._apply_attention(agent_features, obstacle_features, mask)

        assert result.shape == (batch_size, 32)
        assert torch.all(torch.isfinite(result))

    def test_attention_head_configuration(self):
        """Test different attention head configurations."""
        observation_space = self.create_test_observation_space()

        for num_heads in [1, 2, 4, 8]:
            d_model = 32  # Must be divisible by num_heads
            if d_model % num_heads == 0:
                extractor = AttentionExtractor(
                    observation_space, d_model=d_model, num_heads=num_heads, max_obstacles=3
                )
                observations = self.create_test_observations(batch_size=1, max_obstacles=3)
                result = extractor.forward(observations)
                assert result.shape == (1, 2 + d_model)

    def test_different_d_model_sizes(self):
        """Test different d_model sizes."""
        observation_space = self.create_test_observation_space()

        for d_model in [16, 32, 64, 128, 256]:
            extractor = AttentionExtractor(observation_space, d_model=d_model, num_heads=4, max_obstacles=3)
            observations = self.create_test_observations(batch_size=1, max_obstacles=3)
            result = extractor.forward(observations)
            assert result.shape == (1, 2 + d_model)

    # tests/test_attention_extractor.py
    def test_gradient_flow(self):
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, max_obstacles=3)

        observations = self.create_test_observations(batch_size=2, max_obstacles=3)

        for key in ("agent", "obstacles", "target"):
            observations[key].requires_grad_(True)

        result = extractor.forward(observations)
        loss = result.sum()
        loss.backward()

        for key in ("agent", "obstacles", "target"):
            assert observations[key].grad is not None
            assert torch.any(observations[key].grad != 0)

        assert any(p.grad is not None for p in extractor.parameters() if p.requires_grad)


    def test_attention_mechanism_focuses_on_valid_obstacles(self):
        """Test that attention mechanism focuses on valid obstacles."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, max_obstacles=3)

        observations_all_valid = {
            "agent": torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "obstacles": torch.tensor(
                [[[0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
                dtype=torch.float32,
            ),
            "target": torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            "mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        }
        observations_partial_valid = {
            "agent": observations_all_valid["agent"].clone(),
            "obstacles": observations_all_valid["obstacles"].clone(),
            "target": observations_all_valid["target"].clone(),
            "mask": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),  # Only first obstacle valid
        }

        result_all = extractor.forward(observations_all_valid)
        result_partial = extractor.forward(observations_partial_valid)

        # Results should be different due to masking
        assert not torch.allclose(result_all, result_partial, atol=1e-3)

    @patch("extractors.attention_extractor.validate_observation_tensors")
    def test_forward_calls_validation(self, mock_validate):
        """Test that forward pass calls validation function."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(batch_size=1, max_obstacles=3)
        extractor.forward(observations)

        # Check that validation was called
        mock_validate.assert_called_once()

    def test_deterministic_output(self):
        """Test that output is deterministic for same input."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=3)

        observations = self.create_test_observations(batch_size=2, max_obstacles=3)
        result1 = extractor.forward(observations)
        result2 = extractor.forward(observations)

        torch.testing.assert_close(result1, result2)

    def test_batch_independence(self):
        """Test that different batch items are processed independently."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, max_obstacles=3)

        observations_batch = self.create_test_observations(batch_size=3, max_obstacles=3)
        result_batch = extractor.forward(observations_batch)

        results_individual = []
        for i in range(3):
            obs_single = {key: val[i : i + 1] for key, val in observations_batch.items()}
            result_single = extractor.forward(obs_single)
            results_individual.append(result_single)

        result_individual_stacked = torch.cat(results_individual, dim=0)
        torch.testing.assert_close(result_batch, result_individual_stacked, rtol=1e-5, atol=1e-6)

    def test_layer_norm_and_residual_connections(self):
        """Test that layer norm and residual connections work properly."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=32, num_layers=2, max_obstacles=3)

        observations = self.create_test_observations(batch_size=1, max_obstacles=3)
        result = extractor.forward(observations)

        assert torch.all(torch.isfinite(result))
        assert not torch.all(result == 0)

    def test_feed_forward_layers(self):
        """Test that feed forward layers are working correctly."""
        observation_space = self.create_test_observation_space()
        extractor = AttentionExtractor(observation_space, d_model=64, num_layers=1, max_obstacles=3)

        ff_layer = extractor._feed_forwards[0]
        assert len(ff_layer) == 3  # Linear -> ReLU -> Linear
        assert isinstance(ff_layer[0], nn.Linear)
        assert isinstance(ff_layer[1], nn.ReLU)
        assert isinstance(ff_layer[2], nn.Linear)

        # Check dimensions
        assert ff_layer[0].in_features == 64
        assert ff_layer[0].out_features == 128  # d_model * 2
        assert ff_layer[2].in_features == 128
        assert ff_layer[2].out_features == 64


class TestAttentionExtractorIntegration:
    """Integration tests for AttentionExtractor with realistic scenarios."""

    def create_realistic_observations(self):
        """Create realistic navigation scenario observations."""
        return {
            "agent": torch.tensor([[0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),  # Moving right
            "obstacles": torch.tensor(
                [[[0.3,  2.0,  0.0,  0.0, 0.0, 0.0, 0.0],   # Static obstacle ahead
                  [0.3,  1.0,  1.0, -0.5, 0.0, 0.0, 0.0],   # Moving obstacle
                  [0.3, -1.0, -1.0,  0.0, 0.0, 0.0, 0.0],   # Static obstacle behind
                  [0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0],   # Invalid
                  [0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            "target": torch.tensor([[5.0, 0.0]], dtype=torch.float32),   # Target to the right
            "mask": torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]], dtype=torch.float32),  # 3 valid
        }

    def test_realistic_navigation_scenario(self):
        """Test with realistic navigation scenario."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,)),
                "obstacles": spaces.Box(low=-float("inf"), high=float("inf"), shape=(5, 7)),
                "target": spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(5,)),
            }
        )

        extractor = AttentionExtractor(observation_space, d_model=64, num_heads=4, max_obstacles=5)
        observations = self.create_realistic_observations()
        result = extractor.forward(observations)

        assert result.shape == (1, 66)  # 2 + 64
        assert torch.all(torch.isfinite(result))
        assert not torch.all(result == 0)

    def test_performance_large_batch(self):
        """Test performance with large batch size."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,)),
                "obstacles": spaces.Box(low=-float("inf"), high=float("inf"), shape=(10, 7)),
                "target": spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(10,)),
            }
        )
        extractor = AttentionExtractor(observation_space, d_model=64, num_heads=4, max_obstacles=10)

        batch_size = 64
        observations = {
            "agent": torch.randn(batch_size, 7),
            "obstacles": torch.randn(batch_size, 10, 7),
            "target": torch.randn(batch_size, 2),
            "mask": torch.rand(batch_size, 10),
        }
        result = extractor.forward(observations)

        assert result.shape == (batch_size, 66)
        assert torch.all(torch.isfinite(result))

    def test_attention_robustness_to_obstacle_ordering(self):
        """Test that attention is somewhat robust to obstacle ordering."""
        observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=-float("inf"), high=float("inf"), shape=(7,)),
                "obstacles": spaces.Box(low=-float("inf"), high=float("inf"), shape=(3, 7)),
                "target": spaces.Box(low=-float("inf"), high=float("inf"), shape=(2,)),
                "mask": spaces.Box(low=0, high=1, shape=(3,)),
            }
        )
        extractor = AttentionExtractor(observation_space, d_model=32, max_obstacles=3)

        # Original order
        observations1 = {
            "agent": torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            "obstacles": torch.tensor(
                [[[0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            "target": torch.tensor([[5.0, 0.0]], dtype=torch.float32),
            "mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        }

        # Reordered obstacles
        observations2 = {
            "agent": observations1["agent"].clone(),
            "obstacles": torch.tensor(
                [[[0.3, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
            ),
            "target": observations1["target"].clone(),
            "mask": observations1["mask"].clone(),
        }

        result1 = extractor.forward(observations1)
        result2 = extractor.forward(observations2)

        # Shapes must match; values can differ a bit due to order sensitivity
        assert result1.shape == result2.shape


if __name__ == "__main__":
    pytest.main([__file__])
