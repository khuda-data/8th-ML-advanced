import pymunk
import numpy as np
from .entity import Entity
from ..types import CollisionType, Vector2D


class Agent(Entity):
    """Agent class that moves based on external acceleration input"""

    def __init__(
        self,
        space: pymunk.Space,
        position: Vector2D,
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

        # Initialize agent state
        self.reset(position)

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
        """Convert normalized action to acceleration and apply it"""
        max_acceleration = 10.0
        acceleration = Vector2D(
            action[0] * max_acceleration,
            action[1] * max_acceleration,
        )
        self.apply_acceleration(acceleration)

    def update(self, dt: float):
        velocity = self.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.set_velocity(Vector2D(velocity.x * scale, velocity.y * scale))

    def get_state(self) -> np.ndarray:
        pos = self.get_position()
        vel = self.get_velocity()
        return np.array([pos.x, pos.y, vel.x, vel.y], dtype=np.float32)

    def reset(self, position: Vector2D):
        super().reset(position)
        # Agent-specific reset logic can be added here if needed
