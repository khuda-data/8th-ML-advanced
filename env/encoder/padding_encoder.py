import numpy as np
from typing import Dict, Any, List
from .base_encoder import BaseEncoder
from ..entities.entity import Entity


class PaddingEncoder(BaseEncoder):
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
        super().__init__(max_obstacles)

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

    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Encode observation dictionary into a flat vector with padding

        Args:
            observation: Dictionary containing:
                - agent: Agent object
                - obstacles: List of Entity objects
                - target: Vector2D target position

        Returns:
            Encoded observation as numpy array with fixed size
        """
        # Extract components
        agent = observation.get("agent")
        obstacles = observation.get("obstacles", [])
        target = observation.get("target")

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

        return full_encoding

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
