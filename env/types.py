from enum import IntEnum, Enum


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4


class RewardType(float, Enum):
    DISTANCE_ALPHA = 0.1
    TIME_ALPHA = 0.1
    TARGET_REACHED = 100.0
    COLLISION_OCCURRED = 30.0
    TARGET_DESTROYED = 30.0
