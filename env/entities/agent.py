import pymunk
import numpy as np
from .entity import Entity
from ..types import CollisionType, Vector2D


class Agent(Entity):
    """Agent class that moves based on external acceleration input"""

    def __init__(
        self,
        radius: float = 0.5,
        mass: float = 1.0,
        max_force: float = 100.0,
        max_acceleration: float = 10.0,
        world_size: float = 20.0,
    ):
        super().__init__(
            radius=radius,
            mass=mass,
            color=(0, 100, 255),
            collision_type=CollisionType.AGENT,
            world_size=world_size,
        )

        self.max_acceleration = max_acceleration
        self.max_force = max_force
        self.max_velocity = 10.0  # m/s

    def apply_acceleration(self, acceleration: Vector2D):
        force_x = self.body.mass * acceleration.x
        force_y = self.body.mass * acceleration.y

        force_magnitude = np.sqrt(force_x**2 + force_y**2)
        if force_magnitude > self.max_force:
            scale = self.max_force / force_magnitude
            force_x *= scale
            force_y *= scale

        self.apply_force(Vector2D(force_x, force_y))

    def apply_action(self, action: np.ndarray):
        self.apply_acceleration(
            Vector2D(
                action[0] * self.max_acceleration,
                action[1] * self.max_acceleration,
            )
        )

    def update(self, delta_time: float):
        # First call parent update to calculate acceleration
        super().update(delta_time)

        # Then apply velocity constraints
        velocity = self.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.set_velocity(Vector2D(velocity.x * scale, velocity.y * scale))
