from enum import IntEnum
from typing import NamedTuple


class Vector2D(NamedTuple):
    x: float
    y: float


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4
