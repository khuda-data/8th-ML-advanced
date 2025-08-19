import numpy as np
import torch
from torch import Tensor
from typing import List, Dict, Any
from stable_baselines3.sac.policies


class PaddingExtractor():
    """
    Encoder that uses padding to handle variable number of obstacles.
    Always encodes a fixed number of obstacles, padding with zeros when fewer obstacles exist.
    """

    def __init__(self, max_obstacles: int = 10):
        """
        Initialize the PaddingEncoder

        Args:
            max_obstacles: Maximum number of obstacles to encode (with padding)
        """
        super().__init__()
        self.max_obstacles = max_obstacles

        # Observation space structure:
        # - Agent: [pos_x, pos_y, vel_x, vel_y] = 4 elements
        # - Target: [pos_x, pos_y] = 2 elements
        # - Obstacles: [pos_x, pos_y, radius] * max_obstacles = 3 * max_obstacles elements
        self._agent_size = 4
        self._target_size = 2
        self._obstacle_size = 3
        self._obstacles_total_size = self._obstacle_size * self.max_obstacles

        self._observation_size = (
            self._agent_size + self._target_size + self._obstacles_total_size
        )

    def get_observation_space_size(self) -> int:
        """
        Get the size of the observation space

        Returns:
            Size of the observation vector (agent + target + padded obstacles)
        """
        return self._observation_size

    def encode(self, observation: Observation) -> Tensor:
        """
        Encode observation into a flat tensor with padding for neural networks

        Args:
            observation: Typed observation containing:
                - agent: Agent object
                - obstacles: List of Entity objects
                - target: Vector2D target position

        Returns:
            Encoded observation as PyTorch tensor ready for agent/critic networks
        """
        # Extract components from typed observation
        agent = observation.agent
        obstacles = observation.obstacles
        target = observation.target

        # Encode agent
        agent_encoding = self._encode_agent(agent)

        # Encode target
        target_encoding = self._encode_target(target)

        # Encode obstacles with padding
        obstacles_encoding = self._encode_obstacles_with_padding(obstacles)

        # Concatenate all encodings
        full_encoding = np.concatenate(
            [agent_encoding, target_encoding, obstacles_encoding]
        )

        # Convert to PyTorch tensor for neural network input
        return torch.tensor(full_encoding, dtype=torch.float32)

    def _encode_agent(self, agent: Agent) -> np.ndarray:
        """
        Encode agent state

        Args:
            agent: Agent object

        Returns:
            Agent encoding as numpy array [pos_x, pos_y, vel_x, vel_y]
        """
        if agent is None:
            return np.zeros(4, dtype=np.float32)

        position = agent.get_position()
        velocity = agent.get_velocity()

        return np.array(
            [position.x, position.y, velocity.x, velocity.y], dtype=np.float32
        )

    def _encode_obstacles_with_padding(
        self, obstacles: List[Entity]
    ) -> np.ndarray:
        """
        Encode obstacles with padding to reach max_obstacles

        Args:
            obstacles: List of obstacle entities

        Returns:
            Padded obstacles encoding with fixed size
        """
        # Initialize with zeros (padding)
        obstacles_encoding = np.zeros(
            self._obstacles_total_size, dtype=np.float32
        )

        # Encode actual obstacles (up to max_obstacles)
        num_obstacles_to_encode = min(len(obstacles), self.max_obstacles)

        for i in range(num_obstacles_to_encode):
            obstacle = obstacles[i]
            obstacle_encoding = self._encode_obstacle(obstacle)

            # Place encoding at the correct position
            start_idx = i * self._obstacle_size
            end_idx = start_idx + self._obstacle_size
            obstacles_encoding[start_idx:end_idx] = obstacle_encoding

        return obstacles_encoding

    def get_encoding_info(self) -> Dict[str, Any]:
        """
        Get information about the encoding structure

        Returns:
            Dictionary with encoding structure information
        """
        return {
            "total_size": self._observation_size,
            "agent_size": self._agent_size,
            "target_size": self._target_size,
            "obstacles_total_size": self._obstacles_total_size,
            "obstacle_size": self._obstacle_size,
            "max_obstacles": self.max_obstacles,
            "structure": {
                "agent": f"[0:{self._agent_size}] - pos_x, pos_y, vel_x, vel_y",
                "target": f"[{self._agent_size}:{self._agent_size + self._target_size}] - pos_x, pos_y",
                "obstacles": f"[{self._agent_size + self._target_size}:{self._observation_size}] - (pos_x, pos_y, radius) * {self.max_obstacles}",
            },
        }

    def _encode_target(self, target: Vector2D) -> np.ndarray:
        """
        Encode target position

        Args:
            target: Target position

        Returns:
            Target encoding as numpy array [pos_x, pos_y]
        """
        return np.array([target.x, target.y], dtype=np.float32)

    def _encode_obstacle(self, obstacle: Entity) -> np.ndarray:
        """
        Encode single obstacle state

        Args:
            obstacle: Obstacle entity

        Returns:
            Obstacle encoding as numpy array [pos_x, pos_y, radius]
        """
        if obstacle is None:
            return np.zeros(3, dtype=np.float32)

        position = obstacle.get_position()

        return np.array(
            [position.x, position.y, obstacle.radius], dtype=np.float32
        )
