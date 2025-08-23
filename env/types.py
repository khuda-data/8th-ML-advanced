from enum import IntEnum


class CollisionType(IntEnum):
    ENTITY = 1
    AGENT = 2
    OBSTACLE = 3
    WALL = 4  # 벽을 위한 새로운 충돌 유형 추가
