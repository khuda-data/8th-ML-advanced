import numpy as np
import random
from .entity import Entity
from ..types import CollisionType, Vector2D


class StableObstacle(Entity):
    """Obstacle that moves in a constant direction with constant speed"""

    def __init__(
        self,
        position: Vector2D,
        radius: float = 0.3,
        mass: float = 1.0,
        speed: float = 2.0,
    ):
        super().__init__(
            position,
            radius,
            mass,
            color=(255, 100, 100),
            collision_type=CollisionType.OBSTACLE,
        )

        self.speed = speed

    def update(self, dt: float):
        super().update(dt)

        # Ensure velocity remains constant (in case of physics interference)
        current_velocity = self.get_velocity()
        current_speed = np.sqrt(current_velocity.x**2 + current_velocity.y**2)

        if (
            abs(current_speed - self.speed) > 0.01
        ):  # Small tolerance for floating point
            if current_speed > 0:
                scale = self.speed / current_speed
                self.set_velocity(
                    Vector2D(
                        current_velocity.x * scale, current_velocity.y * scale
                    )
                )

    def reset(self):
        super().reset()

        angle = random.uniform(0, 2 * np.pi)
        direction = Vector2D(np.cos(angle), np.sin(angle))

        direction_magnitude = np.sqrt(direction.x**2 + direction.y**2)
        if direction_magnitude > 0:
            normalized_direction = Vector2D(
                direction.x / direction_magnitude,
                direction.y / direction_magnitude,
            )
            velocity = Vector2D(
                normalized_direction.x * self.speed,
                normalized_direction.y * self.speed,
            )
            self.set_velocity(velocity)
