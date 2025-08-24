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
        self.speed = speed

        super().__init__(
            radius=radius,
            mass=mass,
            color=(255, 100, 100),
            collision_type=CollisionType.OBSTACLE,
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
