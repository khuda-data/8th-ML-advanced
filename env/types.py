from enum import IntEnum, Enum


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4


class RewardType(float, Enum):
    TIME_ALPHA = 0.5
    TARGET_REACHED = 30.0
    COLLISION_OCCURRED = 30.0
    TARGET_DESTROYED = 60.0
    STAGNATION_PENALTY_MAX = 5.0
    STAGNATION_PENALTY_DECAY = 70.0
    DIRECTION_SENSITIVITY_K = 10.0