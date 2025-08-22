"""Unit tests for KFEnv class"""

import pytest
import numpy as np
import gymnasium as gym
from env.kf_env import KFEnv


class TestKFEnv:
    """Test cases for KFEnv class"""

    def test_init_default_values(self):
        """Test environment initialization with default values"""
        env = KFEnv()

        assert env.world_size == 20.0
        assert env.max_obstacles == 10
        assert env.target_radius == 1.0
        assert env.render_mode is None

        # Check action space
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)

        # Check observation space
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert "agent" in env.observation_space.spaces
        assert "obstacles" in env.observation_space.spaces
        assert "target" in env.observation_space.spaces
        assert "mask" in env.observation_space.spaces

    def test_init_custom_values(self):
        """Test environment initialization with custom values"""
        env = KFEnv(
            render_mode="human",
            world_size=30.0,
            max_obstacles=5,
            target_radius=2.0,
        )

        assert env.render_mode == "human"
        assert env.world_size == 30.0
        assert env.max_obstacles == 5
        assert env.target_radius == 2.0

    def test_reset(self):
        """Test environment reset functionality"""
        env = KFEnv(max_obstacles=3)

        observation, info = env.reset()

        # Check observation structure
        assert isinstance(observation, dict)
        assert "agent" in observation
        assert "obstacles" in observation
        assert "target" in observation
        assert "mask" in observation

        # Check observation shapes
        assert observation["agent"].shape == (7,)
        assert observation["obstacles"].shape == (3, 7)
        assert observation["target"].shape == (2,)
        assert observation["mask"].shape == (3,)

        # Check data types
        assert observation["agent"].dtype == np.float32
        assert observation["obstacles"].dtype == np.float32
        assert observation["target"].dtype == np.float32
        assert observation["mask"].dtype == np.float32

    def test_step_valid_action(self):
        """Test step with valid action"""
        env = KFEnv()
        env.reset()

        action = np.array([0.5, -0.3], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        # Check return types
        assert isinstance(observation, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_extreme_actions(self):
        """Test step with extreme action values"""
        env = KFEnv()
        env.reset()

        # Test maximum action
        action = np.array([1.0, 1.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        # Test minimum action
        action = np.array([-1.0, -1.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        # Should not raise exceptions

    def test_reward_structure(self):
        """Test reward calculation structure"""
        env = KFEnv()
        env.reset()

        action = np.array([0.0, 0.0], dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)

        # Reward should be a numeric value
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)

    def test_termination_conditions(self):
        """Test environment termination conditions"""
        env = KFEnv(target_radius=10.0)  # Large target for easier reaching
        env.reset()

        # Run several steps
        for _ in range(10):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    def test_observation_bounds(self):
        """Test observation values are within reasonable bounds"""
        env = KFEnv()
        observation, _ = env.reset()

        # Agent position should be within world bounds
        agent_pos = observation["agent"][1:3]  # x, y position
        assert -env.world_size <= agent_pos[0] <= env.world_size
        assert -env.world_size <= agent_pos[1] <= env.world_size

        # Target position should be within world bounds
        target_pos = observation["target"]
        assert -env.world_size <= target_pos[0] <= env.world_size
        assert -env.world_size <= target_pos[1] <= env.world_size

        # Mask values should be 0 or 1
        mask = observation["mask"]
        assert np.all((mask == 0) | (mask == 1))

    def test_multiple_resets(self):
        """Test multiple environment resets"""
        env = KFEnv()

        observations = []
        for _ in range(3):
            observation, _ = env.reset()
            observations.append(observation)

        # Check that resets produce different initial states
        # (with high probability due to randomization)
        agent_positions = [obs["agent"][1:3] for obs in observations]

        # At least some positions should be different
        unique_positions = set(tuple(pos) for pos in agent_positions)
        assert len(unique_positions) > 1 or len(agent_positions) == 1

    def test_action_space_sampling(self):
        """Test action space sampling"""
        env = KFEnv()

        for _ in range(10):
            action = env.action_space.sample()
            assert action.shape == (2,)
            assert -1.0 <= action[0] <= 1.0
            assert -1.0 <= action[1] <= 1.0

    def test_obstacle_mask_consistency(self):
        """Test obstacle mask consistency with obstacle data"""
        env = KFEnv(max_obstacles=5)
        observation, _ = env.reset()

        mask = observation["mask"]
        obstacles = observation["obstacles"]

        # Where mask is 0, obstacle data should be zeros
        for i in range(env.max_obstacles):
            if mask[i] == 0:
                assert np.allclose(obstacles[i], 0.0)

    def test_close_method(self):
        """Test environment close method"""
        env = KFEnv(render_mode="human")
        env.reset()

        # Should not raise exceptions
        env.close()

    def test_render_without_mode(self):
        """Test render method without render mode"""
        env = KFEnv()
        env.reset()

        # Should return None when no render mode is set
        result = env.render()
        assert result is None
