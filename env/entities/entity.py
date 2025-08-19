import pymunk
import pygame
from typing import Tuple, Optional
from ..types import CollisionType, Vector2D


class Entity:
    """Base entity class with physics body. 1 unit = 1 meter."""

    def __init__(
        self,
        radius: float = 1.0,
        mass: float = 1.0,
        color: Tuple[int, int, int] = (100, 100, 100),
        collision_type: CollisionType = CollisionType.ENTITY,
        world_size: float = 20.0,
    ):
        self.space: Optional[pymunk.Space] = None
        self.radius = radius
        self.color = color
        self.mass = mass
        self.collision_type = collision_type
        self.world_size = world_size

        # Track previous velocity for acceleration calculation
        self.previous_velocity = Vector2D(0, 0)
        self.acceleration = Vector2D(0, 0)

        # Create physics body and shape but don't add to space yet
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.7
        self.shape.collision_type = collision_type

        self.reset()

    def set_space(self, space: pymunk.Space) -> None:
        """
        Set the physics space for this entity

        Args:
            space: Pymunk space to add this entity to
        """
        # Remove from old space if exists
        if self.space is not None:
            self.unset_space()

        self.space = space
        if self.space is not None:
            self.space.add(self.body, self.shape)

    def get_position(self) -> Vector2D:
        return Vector2D(self.body.position.x, self.body.position.y)

    def get_velocity(self) -> Vector2D:
        return Vector2D(self.body.velocity.x, self.body.velocity.y)

    def get_acceleration(self) -> Vector2D:
        return self.acceleration

    def set_position(self, position: Vector2D):
        self.body.position = position.x, position.y

    def set_velocity(self, velocity: Vector2D):
        self.body.velocity = velocity.x, velocity.y

    def apply_force(self, force: Vector2D):
        self.body.apply_force_at_local_point((force.x, force.y), (0, 0))

    def apply_impulse(self, impulse: Vector2D):
        self.body.apply_impulse_at_local_point((impulse.x, impulse.y), (0, 0))

    def render(
        self, screen: pygame.Surface, scale: float, offset: float
    ) -> None:
        """Render entity on pygame screen with scaling and offset"""
        pos = self.get_position()
        screen_x = int(pos.x * scale + offset)
        screen_y = int(pos.y * scale + offset)
        radius = int(self.radius * scale)

        pygame.draw.circle(screen, self.color, (screen_x, screen_y), radius)
        pygame.draw.circle(screen, (0, 0, 0), (screen_x, screen_y), radius, 2)

    def update(self, delta_time: float):
        """Update entity and calculate acceleration"""
        current_velocity = self.get_velocity()

        # Calculate acceleration based on velocity change
        self.acceleration = Vector2D(
            (
                (current_velocity.x - self.previous_velocity.x) / delta_time
                if delta_time > 0
                else 0
            ),
            (
                (current_velocity.y - self.previous_velocity.y) / delta_time
                if delta_time > 0
                else 0
            ),
        )

        # Store current velocity for next update
        self.previous_velocity = current_velocity

    def unset_space(self):
        """Remove this entity from its physics space"""
        if self.space is not None and self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)
            self.space = None

    def reset(self, world_size: float = None):
        """
        Reset entity to a random position within world bounds

        Args:
            world_size: Size of the world, or None to use entity's world_size
        """
        if world_size is not None:
            self.world_size = world_size

        # Generate random position within world bounds
        half_size = (
            self.world_size / 2 - self.radius
        )  # Leave margin for entity radius
        import random

        x = random.uniform(-half_size, half_size)
        y = random.uniform(-half_size, half_size)
        position = Vector2D(x, y)

        self.set_position(position)
        self.set_velocity(Vector2D(0, 0))
        self.body.angular_velocity = 0

        # Reset acceleration tracking
        self.previous_velocity = Vector2D(0, 0)
        self.acceleration = Vector2D(0, 0)
