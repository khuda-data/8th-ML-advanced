import pymunk
import pygame
from typing import Tuple
from .types import CollisionType


class Entity:
    """Base entity class with physics body. 1 unit = 1 meter."""

    def __init__(
        self,
        space: pymunk.Space,
        position: Tuple[float, float],
        radius: float = 1.0,
        mass: float = 1.0,
        color: Tuple[int, int, int] = (100, 100, 100),
        collision_type: CollisionType = CollisionType.ENTITY,
    ):
        self.space = space
        self.radius = radius
        self.color = color

        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.friction = 0.7
        self.shape.collision_type = collision_type

        self.space.add(self.body, self.shape)

    def get_position(self) -> Tuple[float, float]:
        return self.body.position.x, self.body.position.y

    def get_velocity(self) -> Tuple[float, float]:
        return self.body.velocity.x, self.body.velocity.y

    def set_position(self, position: Tuple[float, float]):
        self.body.position = position

    def set_velocity(self, velocity: Tuple[float, float]):
        self.body.velocity = velocity

    def apply_force(self, force: Tuple[float, float]):
        self.body.apply_force_at_local_point(force, (0, 0))

    def apply_impulse(self, impulse: Tuple[float, float]):
        self.body.apply_impulse_at_local_point(impulse, (0, 0))

    def render(self, screen: pygame.Surface):
        """Render entity on pygame screen"""
        pos_x = int(self.body.position.x)
        pos_y = int(self.body.position.y)
        radius_pixels = int(self.radius * 50)  # 1m = 50 pixels

        pygame.draw.circle(screen, self.color, (pos_x, pos_y), radius_pixels)
        pygame.draw.circle(screen, (0, 0, 0), (pos_x, pos_y), radius_pixels, 2)

    def update(self, dt: float):
        """Override for entity-specific updates"""
        pass

    def remove_from_space(self):
        if self.body in self.space.bodies:
            self.space.remove(self.body, self.shape)
