import pymunk
import numpy as np
from typing import Tuple
from .entity import Entity
from .types import CollisionType


class Agent(Entity):
    """Agent class that moves based on external acceleration input"""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        radius: float = 0.5,
        mass: float = 1.0,
        max_force: float = 100.0,
    ):
        super().__init__(
            space,
            position,
            radius,
            mass,
            color=(0, 100, 255),
            collision_type=CollisionType.AGENT,
        )

        self.max_force = max_force
        self.max_velocity = 10.0  # m/s

    def apply_acceleration(self, acceleration: Tuple[float, float]):
        # F = ma
        force_x = self.body.mass * acceleration[0]
        force_y = self.body.mass * acceleration[1]

        force_magnitude = np.sqrt(force_x**2 + force_y**2)
        if force_magnitude > self.max_force:
            scale = self.max_force / force_magnitude
            force_x *= scale
            force_y *= scale

        self.apply_force((force_x, force_y))

    def apply_action(self, action: np.ndarray):
        """Convert normalized action to acceleration and apply it"""
        max_acceleration = 10.0
        acceleration = (
            action[0] * max_acceleration,
            action[1] * max_acceleration,
        )
        self.apply_acceleration(acceleration)

    def update(self, dt: float):
        velocity = self.get_velocity()
        speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)

        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.set_velocity((velocity[0] * scale, velocity[1] * scale))

    def get_state(self) -> np.ndarray:
        pos = self.get_position()
        vel = self.get_velocity()
        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float32)

    def reset(self, position: Tuple[float, float]):
        self.set_position(position)
        self.set_velocity((0, 0))
        self.body.angular_velocity = 0
