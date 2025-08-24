"""Unit tests for Entity base class"""

import pymunk
from pygame import Vector2
from ..entity import Entity
from ...types import CollisionType


class TestEntity:
    """Test cases for Entity class"""

    def test_init_default_values(self):
        """Test entity initialization with default values"""
        entity = Entity()

        assert entity.radius == 1.0
        assert entity.mass == 1.0
        assert entity.color == (100, 100, 100)
        assert entity.collision_type == CollisionType.ENTITY
        assert entity.space is None
        assert isinstance(entity.body, pymunk.Body)
        assert isinstance(entity.shape, pymunk.Circle)

    def test_init_custom_values(self):
        """Test entity initialization with custom values"""
        entity = Entity(
            radius=2.0,
            mass=5.0,
            color=(255, 0, 0),
            collision_type=CollisionType.AGENT,
        )

        assert entity.radius == 2.0
        assert entity.mass == 5.0
        assert entity.color == (255, 0, 0)
        assert entity.collision_type == CollisionType.AGENT

    def test_set_position(self):
        """Test setting entity position"""
        entity = Entity()
        position = Vector2(10.0, 5.0)

        entity.set_position(position)
        result = entity.get_position()

        assert abs(result.x - 10.0) < 0.001
        assert abs(result.y - 5.0) < 0.001

    def test_set_velocity(self):
        """Test setting entity velocity"""
        entity = Entity()
        velocity = Vector2(3.0, 4.0)

        entity.set_velocity(velocity)
        result = entity.get_velocity()

        assert abs(result.x - 3.0) < 0.001
        assert abs(result.y - 4.0) < 0.001

    def test_apply_force(self):
        """Test applying force to entity"""
        entity = Entity(mass=1.0)
        force = Vector2(10.0, 0.0)

        entity.apply_force(force)
        # Force should be applied to body
        assert hasattr(entity.body, "force")

    def test_set_space(self):
        """Test adding entity to physics space"""
        entity = Entity()
        space = pymunk.Space()

        entity.set_space(space)

        assert entity.space == space
        assert entity.body in space.bodies
        assert entity.shape in space.shapes

    def test_unset_space(self):
        """Test removing entity from physics space"""
        entity = Entity()
        space = pymunk.Space()

        entity.set_space(space)
        entity.unset_space()

        assert entity.space is None
        assert entity.body not in space.bodies
        assert entity.shape not in space.shapes

    def test_update(self):
        """Test entity update method"""
        entity = Entity()
        dt = 0.016  # 60 FPS

        # Should not raise any exceptions
        entity.update(dt)

    def test_reset(self):
        """Test entity reset functionality"""
        entity = Entity()
        entity.set_position(Vector2(10.0, 10.0))
        entity.set_velocity(Vector2(5.0, 5.0))

        entity.reset()

        position = entity.get_position()
        velocity = entity.get_velocity()

        assert abs(position.x) < 0.001
        assert abs(position.y) < 0.001
        assert abs(velocity.x) < 0.001
        assert abs(velocity.y) < 0.001

    def test_distance_to(self):
        """Test distance calculation between entities"""
        entity1 = Entity()
        entity2 = Entity()

        entity1.set_position(Vector2(0.0, 0.0))
        entity2.set_position(Vector2(3.0, 4.0))

        distance = entity1.distance_to(entity2)

        assert abs(distance - 5.0) < 0.001

    def test_render_without_screen(self):
        """Test render method without screen (should not crash)"""
        entity = Entity()

        # Should not raise exceptions when screen is None
        entity.render(None, 1.0, Vector2(0, 0))
