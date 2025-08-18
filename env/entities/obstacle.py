import pymunk
import numpy as np
import random
from .entity import Entity
from ..types import CollisionType, Vector2D


class Obstacle(Entity):
    """Obstacle that moves in random straight lines"""

    def __init__(
        self,
        space: pymunk.Space,
        position: Vector2D,
        radius: float = 0.3,
        mass: float = 1.0,
        speed: float = 2.0,
    ):
        super().__init__(
            space,
            position,
            radius,
            mass,
            color=(255, 100, 100),
            collision_type=CollisionType.OBSTACLE,
        )

        self.speed = speed
        self.direction_change_timer = 0
        self.direction_change_interval = random.uniform(2.0, 5.0)

        # Initialize obstacle state
        self.reset(position)

    def _set_random_direction(self):
        angle = random.uniform(0, 2 * np.pi)
        velocity_x = self.speed * np.cos(angle)
        velocity_y = self.speed * np.sin(angle)
        self.set_velocity(Vector2D(velocity_x, velocity_y))

    def update(self, dt: float):
        self.direction_change_timer += dt

        if self.direction_change_timer >= self.direction_change_interval:
            self._set_random_direction()
            self.direction_change_timer = 0
            self.direction_change_interval = random.uniform(2.0, 5.0)

        current_velocity = self.get_velocity()
        current_speed = np.sqrt(current_velocity.x**2 + current_velocity.y**2)

        if current_speed > 0:
            scale = self.speed / current_speed
            self.set_velocity(
                Vector2D(current_velocity.x * scale, current_velocity.y * scale)
            )

    def reset(self, position: Vector2D):
        super().reset(position)
        self.direction_change_timer = 0
        self.direction_change_interval = random.uniform(2.0, 5.0)
        self._set_random_direction()
