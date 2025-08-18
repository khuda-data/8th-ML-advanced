from enum import IntEnum
from typing import NamedTuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .entities.agent import Agent
    from .entities.entity import Entity


class Vector2D(NamedTuple):
    x: float
    y: float


class Observation(NamedTuple):
    """
    Typed observation structure for environment observations

    This ensures type safety and consistency between env and encoders
    """

    agent: "Agent"
    obstacles: List["Entity"]
    target: Vector2D


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4
