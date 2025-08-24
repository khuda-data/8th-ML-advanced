import torch
import torch.nn as nn
from typing import Dict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .utils import (
    extract_agent_features,
    extract_target_relative_features,
    extract_obstacle_relative_features,
    get_feature_dimensions,
    validate_observation_tensors,
)


class AttentionExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using multi-head attention mechanism for obstacle processing.

    This extractor uses attention mechanisms to dynamically focus on relevant obstacles
    rather than processing all obstacles equally. The agent state serves as the query,
    while obstacle features serve as keys and values. This allows the model to
    selectively attend to obstacles that are most relevant for decision making.

    The attention mechanism is particularly effective when:
    - The number of obstacles varies significantly
    - Some obstacles are more important than others for navigation
    - Spatial relationships between agent and obstacles matter

    Architecture:
    1. Project agent features to d_model dimensions (query)
    2. Project obstacle features to d_model dimensions (keys and values)
    3. Apply multi-head attention across multiple layers
    4. Concatenate final agent representation with target features

    Args:
        observation_space: The observation space from the Gymnasium environment
        d_model: Dimension of the attention model (default: 64)
        num_heads: Number of attention heads (default: 4)
        num_layers: Number of attention layers (default: 1)
        max_obstacles: Maximum number of obstacles (default: 10)
        include_acceleration: Whether to include acceleration features (default: False)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        max_obstacles: int = 10,
        include_acceleration: bool = False,
        **kwargs,
    ):
        self._d_model = d_model
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._max_obstacles = max_obstacles
        self._include_acceleration = include_acceleration

        # Get standard feature dimensions
        (
            self._agent_size,
            self._target_size,
            self._obstacle_size,
            _,  # obstacles_total_size not needed for attention
        ) = get_feature_dimensions(max_obstacles, include_acceleration)

        features_dim = self._target_size + d_model
        super().__init__(observation_space, features_dim)

        self._initial_agent_projection = nn.Linear(self._agent_size, d_model)

        self._q_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_layers)]
        )
        self._k_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )
        self._v_projections = nn.ModuleList(
            [nn.Linear(self._obstacle_size, d_model) for _ in range(num_layers)]
        )

        self._mha_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model, num_heads=num_heads, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        self._layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        self._feed_forwards = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(num_layers)
            ]
        )

        self._output_projection = nn.Linear(d_model, d_model)

    def _create_attention_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Create an attention mask for PyTorch MultiheadAttention mechanism.

        This function converts obstacle validity masks into the format expected
        by PyTorch's MultiheadAttention layers. The attention mechanism uses
        boolean masks where True indicates positions that should be ignored
        (masked out) during attention computation.

        Args:
            mask: Tensor of shape [batch_size, max_obstacles] containing obstacle
                validity indicators where 1.0 means valid obstacle and 0.0
                means invalid/padding obstacle.

        Returns:
            Tensor of shape [batch_size, max_obstacles] with boolean values where
            True indicates invalid obstacles that should be ignored in attention
            computation and False indicates valid obstacles that should participate
            in attention.

        Note:
            This creates an inverted mask because MultiheadAttention expects
            True for positions to ignore, while our obstacle masks use 1 for
            valid positions.
        """
        return mask == 0

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract features using multi-head attention mechanism.

        This method processes observations by applying attention between the agent
        and all obstacles. The agent state is used as the query while obstacle
        features serve as keys and values, allowing the model to focus on the
        most relevant obstacles for navigation decisions.

        Args:
            observations: Dictionary containing:
                - "agent": Tensor [batch_size, 7] with agent state
                - "obstacles": Tensor [batch_size, max_obstacles, 7] with obstacle states
                - "target": Tensor [batch_size, 2] with target position
                - "mask": Tensor [batch_size, max_obstacles] with obstacle validity

        Returns:
            Tensor of shape [batch_size, target_size + d_model] containing
            concatenated target features and attention-processed agent features
        """
        agent_data = observations["agent"]
        obstacles_data = observations["obstacles"]
        target_data = observations["target"]
        mask = observations["mask"]

        validate_observation_tensors(
            agent_data, obstacles_data, target_data, mask, self._max_obstacles
        )

        agent_features = extract_agent_features(
            agent_data, self._include_acceleration
        )

        target_features = extract_target_relative_features(
            agent_data, target_data
        )

        obstacle_features = extract_obstacle_relative_features(
            agent_data, obstacles_data, mask, self._include_acceleration
        )

        attended_features = self._apply_attention(
            agent_features, obstacle_features, mask
        )

        return torch.cat([target_features, attended_features], dim=1)

    def _apply_attention(
        self,
        agent_features: torch.Tensor,
        obstacle_features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply multi-layer attention mechanism between agent and obstacles.

        This private method implements the core attention computation where the
        agent features serve as queries and obstacle features serve as both
        keys and values. Multiple attention layers with residual connections
        and layer normalization are applied to create rich representations.

        Args:
            agent_features: Tensor [batch_size, agent_size] with agent state features
            obstacle_features: Tensor [batch_size, max_obstacles, obstacle_size]
                              with relative obstacle features
            mask: Tensor [batch_size, max_obstacles] indicating valid obstacles

        Returns:
            Tensor [batch_size, d_model] containing the final attended agent representation
            that incorporates information from all relevant obstacles
        """
        attended_output = self._initial_agent_projection(
            agent_features
        ).unsqueeze(1)

        layers = zip(
            range(self._num_layers),
            self._q_projections,
            self._k_projections,
            self._v_projections,
            self._mha_layers,
            self._layer_norms,
            self._feed_forwards,
        )

        for _, q_proj, k_proj, v_proj, mha, ln, ff in layers:
            q = q_proj(attended_output.squeeze(1)).unsqueeze(1)
            k = k_proj(obstacle_features)
            v = v_proj(obstacle_features)

            residual = attended_output

            attention_mask = self._create_attention_mask(mask)

            mha_output, _ = mha(
                query=q,
                key=k,
                value=v,
                key_padding_mask=attention_mask,
                need_weights=False,
            )

            attended_output = ln(residual + mha_output)

            residual = attended_output
            ff_output = ff(attended_output)
            attended_output = residual + ff_output

        attended_output = attended_output.squeeze(1)
        return self._output_projection(attended_output)
