import pymunk
import numpy as np
from pygame import Vector2
from .entity import Entity
from ..types import CollisionType


class Agent(Entity):
    """Agent class that moves based on external acceleration input"""

    def __init__(
        self,
        radius: float = 0.5,
        mass: float = 1.0,
        max_force: float = 100.0,
        max_acceleration: float = 10.0,
    ):
        super().__init__(
            radius=radius,
            mass=mass,
            color=(0, 100, 255),
            collision_type=CollisionType.AGENT,
        )

        self.max_acceleration = max_acceleration
        self.max_force = max_force
        self.max_velocity = 10.0

        OBSTACLE_CAT = 0b0010
        AGENT_CAT    = 0b0100
        self.shape.filter = pymunk.ShapeFilter(
            categories=AGENT_CAT,
            mask=OBSTACLE_CAT | AGENT_CAT  # 벽 비트 제외
        )

    def apply_acceleration(self, acceleration: Vector2):
        force_x = self.body.mass * acceleration.x
        force_y = self.body.mass * acceleration.y

        force_magnitude = np.sqrt(force_x**2 + force_y**2)
        if force_magnitude > self.max_force:
            scale = self.max_force / force_magnitude
            force_x *= scale
            force_y *= scale

        self.apply_force(Vector2(force_x, force_y))

    def apply_action(self, action: np.ndarray):
        self.apply_acceleration(
            Vector2(
                action[0] * self.max_acceleration,
                action[1] * self.max_acceleration,
            )
        )

    def update(self, dt: float):
        super().update(dt)

        # Then apply velocity constraints
        velocity = self.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.set_velocity(Vector2(velocity.x * scale, velocity.y * scale))
