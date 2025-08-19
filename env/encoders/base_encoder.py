from abc import ABC, abstractmethod
import numpy as np
from ..types import Observation
from torch import Tensor


class BaseEncoder(ABC):
    """Base class for observation encoders"""

    def __init__(self):
        """
        Initialize the encoder
        """
        pass

    @abstractmethod
    def get_observation_space_size(self) -> int:
        """
        Get the size of the observation space

        Returns:
            Size of the observation vector
        """
        pass

    @abstractmethod
    def encode(self, observation: Observation) -> Tensor:
        """
        Encode observation into a flat vector

        Args:
            observation: Typed observation containing:
                - agent: Agent object
                - obstacles: List of Entity objects
                - target: Vector2D target position

        Returns:
            Encoded observation as numpy array
        """
        pass
