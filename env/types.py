from enum import IntEnum


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4


class EntityType(IntEnum):
    WALL = 1 << 0  # 0b0001
    OBSTACLE = 1 << 1  # 0b0010
    AGENT = 1 << 2  # 0b0100
