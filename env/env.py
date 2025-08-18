import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import numpy as np
import random
from typing import List

from env.types import CollisionType
from .agent import Agent
from .obstacle import Obstacle


class KFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self, render_mode=None, world_size: float = 20.0, num_obstacles: int = 5
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.num_obstacles = num_obstacles

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Agent state (pos_x, pos_y, vel_x, vel_y) + obstacles positions
        obs_dim = 4 + self.num_obstacles * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        pygame.init()
        self.screen_size = 800
        self.screen = pygame.display.set_mode(
            (self.screen_size, self.screen_size)
        )
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity for top-down view

        self.agent: Agent
        self.obstacles: List[Obstacle] = []
        self.target_position = (0, 0)

        self._setup_boundaries()

        self.reset()

    def _setup_boundaries(self):
        """Create world boundaries"""
        half_size = self.world_size / 2
        thickness = 1.0

        # Create static bodies for walls
        walls = [
            # Top
            (0, half_size + thickness / 2, self.world_size, thickness),
            # Bottom
            (0, -half_size - thickness / 2, self.world_size, thickness),
            # Left
            (-half_size - thickness / 2, 0, thickness, self.world_size),
            # Right
            (half_size + thickness / 2, 0, thickness, self.world_size),
        ]

        for x, y, w, h in walls:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = x, y
            shape = pymunk.Poly.create_box(body, (w, h))
            shape.friction = 0.7
            shape.collision_type = CollisionType.WALL
            self.space.add(body, shape)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Clear existing entities
        if self.agent:
            self.agent.remove_from_space()
        for obstacle in self.obstacles:
            obstacle.remove_from_space()

        # Reset agent
        agent_pos = self._get_random_position()
        self.agent = Agent(self.space, agent_pos)

        # Reset obstacles
        self.obstacles = []
        for _ in range(self.num_obstacles):
            obs_pos = self._get_random_position()
            obstacle = Obstacle(self.space, obs_pos)
            self.obstacles.append(obstacle)

        # Set target position
        self.target_position = self._get_random_position()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Apply action to agent
        self.agent.apply_action(action)

        # Update physics
        dt = 1.0 / self.metadata["render_fps"]
        self.space.step(dt)

        # Update entities
        self.agent.update(dt)
        for obstacle in self.obstacles:
            obstacle.update(dt)

        # Calculate reward and check termination
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._check_collision()
        truncated = False
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))  # White background

        # Convert world coordinates to screen coordinates
        scale = self.screen_size / self.world_size
        offset = self.screen_size / 2

        # Draw target
        target_screen_x = int(self.target_position[0] * scale + offset)
        target_screen_y = int(self.target_position[1] * scale + offset)
        pygame.draw.circle(
            self.screen, (0, 255, 0), (target_screen_x, target_screen_y), 20, 3
        )

        # Draw entities
        if self.agent:
            self._render_entity(self.agent, scale, offset)

        for obstacle in self.obstacles:
            self._render_entity(obstacle, scale, offset)

        # Draw boundaries
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            pygame.Rect(0, 0, self.screen_size, self.screen_size),
            3,
        )

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _render_entity(self, entity, scale, offset):
        pos = entity.get_position()
        screen_x = int(pos[0] * scale + offset)
        screen_y = int(pos[1] * scale + offset)
        radius = int(entity.radius * scale)

        pygame.draw.circle(
            self.screen, entity.color, (screen_x, screen_y), radius
        )
        pygame.draw.circle(
            self.screen, (0, 0, 0), (screen_x, screen_y), radius, 2
        )

    def _get_random_position(self):
        half_size = self.world_size / 2 - 2  # Leave margin from walls
        x = random.uniform(-half_size, half_size)
        y = random.uniform(-half_size, half_size)
        return (x, y)

    def _get_obs(self):
        if not self.agent:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        agent_state = self.agent.get_state()

        # Get obstacle positions
        obstacle_positions = []
        for obstacle in self.obstacles:
            pos = obstacle.get_position()
            obstacle_positions.extend([pos[0], pos[1]])

        # Pad if fewer obstacles than expected
        while len(obstacle_positions) < self.num_obstacles * 2:
            obstacle_positions.extend([0.0, 0.0])

        obs = np.concatenate([agent_state, obstacle_positions])
        return obs.astype(np.float32)

    def _calculate_reward(self):
        if not self.agent:
            return 0.0

        agent_pos = self.agent.get_position()
        distance_to_target = np.sqrt(
            (agent_pos[0] - self.target_position[0]) ** 2
            + (agent_pos[1] - self.target_position[1]) ** 2
        )

        # Negative distance as reward (closer is better)
        reward = -distance_to_target * 0.1

        # Bonus for reaching target
        if distance_to_target < 1.0:
            reward += 10.0

        return reward

    def _check_collision(self):
        if not self.agent:
            return False

        agent_pos = self.agent.get_position()

        # Check collision with obstacles
        for obstacle in self.obstacles:
            obs_pos = obstacle.get_position()
            distance = np.sqrt(
                (agent_pos[0] - obs_pos[0]) ** 2
                + (agent_pos[1] - obs_pos[1]) ** 2
            )
            if distance < (self.agent.radius + obstacle.radius):
                return True

        # Check if agent is out of bounds
        half_size = self.world_size / 2
        if abs(agent_pos[0]) > half_size or abs(agent_pos[1]) > half_size:
            return True

        return False

    def _get_info(self):
        return {}

    def close(self):
        pygame.quit()
