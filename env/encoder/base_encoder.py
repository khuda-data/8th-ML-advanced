from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
from ..types import Vector2D
from ..entities.entity import Entity
from ..entities.agent import Agent


class BaseEncoder(ABC):
    """Base class for observation encoders"""

    def __init__(self, max_obstacles: int = 10):
        """
        Initialize the encoder

        Args:
            max_obstacles: Maximum number of obstacles to encode
        """
        self.max_obstacles = max_obstacles

    @abstractmethod
    def get_observation_space_size(self) -> int:
        """
        Get the size of the observation space

        Returns:
            Size of the observation vector
        """
        pass

    @abstractmethod
    def encode(self, observation: Dict[str, Any]) -> np.ndarray:
        """
        Encode observation dictionary into a flat vector

        Args:
            observation: Dictionary containing:
                - agent: Agent object
                - obstacles: List of Entity objects
                - target: Vector2D target position

        Returns:
            Encoded observation as numpy array
        """
        pass

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
