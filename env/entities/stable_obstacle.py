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
        direction: Vector2D = None,
    ):
        super().__init__(
            position,
            radius,
            mass,
            color=(255, 100, 100),
            collision_type=CollisionType.OBSTACLE,
        )

        self.speed = speed
        self.direction = direction

        self.reset(position=position, direction=direction)

    def _initialize_movement(self):
        """Initialize movement with stable direction and speed"""
        if self.direction is None:
            # Set random direction if not specified
            angle = random.uniform(0, 2 * np.pi)
            self.direction = Vector2D(np.cos(angle), np.sin(angle))

        # Normalize direction and apply speed
        direction_magnitude = np.sqrt(self.direction.x**2 + self.direction.y**2)
        if direction_magnitude > 0:
            normalized_direction = Vector2D(
                self.direction.x / direction_magnitude,
                self.direction.y / direction_magnitude,
            )
            velocity = Vector2D(
                normalized_direction.x * self.speed,
                normalized_direction.y * self.speed,
            )
            self.set_velocity(velocity)

    def update(self, delta_time: float):
        """Maintain constant velocity"""
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

    def reset(self, position: Vector2D, direction: Vector2D = None):
        """Reset obstacle to initial state with optional new direction"""
        super().reset(position)
        if direction is not None:
            self.direction = direction
        self._initialize_movement()
