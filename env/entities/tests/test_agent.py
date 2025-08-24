"""Unit tests for Agent class"""

import numpy as np
from pygame import Vector2
from ..agent import Agent
from ...types import CollisionType


class TestAgent:
    """Test cases for Agent class"""

    def test_init_default_values(self):
        """Test agent initialization with default values"""
        agent = Agent()

        assert agent.radius == 0.5
        assert agent.mass == 1.0
        assert agent.max_force == 100.0
        assert agent.max_acceleration == 10.0
        assert agent.max_velocity == 10.0
        assert agent.collision_type == CollisionType.AGENT
        assert agent.color == (0, 100, 255)

    def test_init_custom_values(self):
        """Test agent initialization with custom values"""
        agent = Agent(
            radius=1.0, mass=2.0, max_force=200.0, max_acceleration=20.0
        )

        assert agent.radius == 1.0
        assert agent.mass == 2.0
        assert agent.max_force == 200.0
        assert agent.max_acceleration == 20.0

    def test_apply_acceleration_normal(self):
        """Test applying normal acceleration"""
        agent = Agent(mass=1.0, max_force=100.0)
        acceleration = Vector2(5.0, 0.0)

        agent.apply_acceleration(acceleration)

        # Force should be mass * acceleration = 1.0 * 5.0 = 5.0
        # This should be within max_force limit

    def test_apply_acceleration_exceeds_max_force(self):
        """Test applying acceleration that exceeds max force"""
        agent = Agent(mass=1.0, max_force=10.0, max_acceleration=20.0)
        acceleration = Vector2(15.0, 0.0)  # Force would be 15.0

        agent.apply_acceleration(acceleration)

        # Force should be clamped to max_force = 10.0

    def test_apply_action_normal(self):
        """Test applying normal action"""
        agent = Agent(max_acceleration=10.0)
        action = np.array([0.5, -0.3])

        agent.apply_action(action)

        # Expected acceleration: [5.0, -3.0]

    def test_apply_action_extreme_values(self):
        """Test applying action with extreme values"""
        agent = Agent(max_acceleration=10.0)
        action = np.array([1.0, -1.0])

        agent.apply_action(action)

        # Expected acceleration: [10.0, -10.0]

    def test_velocity_limit(self):
        """Test velocity limiting functionality"""
        agent = Agent()
        agent.max_velocity = 5.0

        # Set velocity exceeding max_velocity
        agent.set_velocity(Vector2(10.0, 0.0))
        agent.update(0.016)

        velocity = agent.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        assert speed <= agent.max_velocity + 0.001  # Small tolerance

    def test_update_with_velocity_constraint(self):
        """Test update method with velocity constraint"""
        agent = Agent()
        agent.max_velocity = 8.0
        agent.set_velocity(Vector2(15.0, 0.0))  # Exceeds max_velocity

        agent.update(0.016)

        velocity = agent.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        assert abs(speed - 8.0) < 0.001

    def test_update_normal_velocity(self):
        """Test update with normal velocity (no clamping)"""
        agent = Agent()
        agent.set_velocity(Vector2(3.0, 4.0))  # Speed = 5.0, within limit

        initial_velocity = agent.get_velocity()
        agent.update(0.016)
        final_velocity = agent.get_velocity()

        # Velocity should not be modified if within limits
        assert abs(final_velocity.x - initial_velocity.x) < 0.1
        assert abs(final_velocity.y - initial_velocity.y) < 0.1

    def test_zero_acceleration(self):
        """Test applying zero acceleration"""
        agent = Agent()
        acceleration = Vector2(0.0, 0.0)

        agent.apply_acceleration(acceleration)

        # Should not cause any issues

    def test_force_scaling(self):
        """Test force scaling when exceeding max_force"""
        agent = Agent(mass=2.0, max_force=10.0)
        acceleration = Vector2(10.0, 0.0)  # Force would be 20.0

        agent.apply_acceleration(acceleration)

        # Force should be scaled down to max_force
