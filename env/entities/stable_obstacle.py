import numpy as np
import random
from pygame import Vector2
from .entity import Entity
from ..types import CollisionType
import pymunk


class StableObstacle(Entity):
    """Obstacle that moves in a constant direction with constant speed"""

    def __init__(
        self,
        radius: float = 0.5,
        mass: float = 1.0,
        speed: float = 2.0,
    ):
        # self.speed를 super().__init__() 호출보다 먼저 정의합니다.
        self.speed = speed

        # 이제 부모 클래스의 생성자를 호출합니다.
        # 부모 생성자 내부에서 reset()이 호출될 때 self.speed가 존재하게 됩니다.
        super().__init__(
            radius=radius,
            mass=mass,
            color=(255, 100, 100),
            collision_type=CollisionType.OBSTACLE,
        )

        self.shape.elasticity = 1.0
        self.shape.friction = 0.0

        WALL_CAT     = 0b0001
        OBSTACLE_CAT = 0b0010
        AGENT_CAT    = 0b0100
        self.shape.filter = pymunk.ShapeFilter(
            categories=OBSTACLE_CAT,
            mask=WALL_CAT | AGENT_CAT | OBSTACLE_CAT
        )

    def update(self, dt: float):
        super().update(dt)

    def reset(self):
        super().reset()

        angle = random.uniform(0, 2 * np.pi)
        direction = Vector2(np.cos(angle), np.sin(angle))

        direction_magnitude = np.sqrt(direction.x**2 + direction.y**2)
        if direction_magnitude > 0:
            normalized_direction = Vector2(
                direction.x / direction_magnitude,
                direction.y / direction_magnitude,
            )
            velocity = Vector2(
                normalized_direction.x * self.speed,
                normalized_direction.y * self.speed,
            )
            self.set_velocity(velocity)