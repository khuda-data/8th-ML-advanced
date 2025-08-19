import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import numpy as np
import random
from typing import List, Optional, Tuple, Dict, Any

from .types import CollisionType, Vector2D
from .entities import Agent, Entity


class KFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: float = 20.0,
        max_obstacles: int = 10,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.max_obstacles = max_obstacles

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Each entity: [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y] = 7 dimensions
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
                ),
                "obstacles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_obstacles, 7),
                    dtype=np.float32,
                ),
                "target": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                ),
                "mask": spaces.Box(
                    low=0, high=1, shape=(self.max_obstacles,), dtype=np.float32
                ),
            }
        )

        pygame.init()
        self.screen_size = 800
        self.screen = pygame.display.set_mode(
            (self.screen_size, self.screen_size)
        )
        self.clock = pygame.time.Clock()

        self.space = pymunk.Space()
        self.space.gravity = (0, 0)  # No gravity for top-down view

        self.agent: Optional[Agent] = None
        self.obstacles: List[Entity] = []
        self.target_position: Vector2D = Vector2D(0, 0)

        # Collision flag for pymunk collision handler
        self.collision_occurred = False

        # Setup collision handlers
        self._setup_collision_handlers()

        self.reset()

    def _get_all_entities(self) -> List[Entity]:
        """Get list of all entities (agent + obstacles)"""
        all_entities = []
        if self.agent is not None:
            all_entities.append(self.agent)
        all_entities.extend(self.obstacles)
        return all_entities

    def _setup_collision_handlers(self) -> None:
        """Setup pymunk collision handlers for different collision types"""
        # Agent-Obstacle collision handler
        handler = self.space.add_collision_handler(
            CollisionType.AGENT, CollisionType.OBSTACLE
        )
        handler.begin = self._on_agent_collision

        # Agent-Entity collision handler (for generic entities)
        handler2 = self.space.add_collision_handler(
            CollisionType.AGENT, CollisionType.ENTITY
        )
        handler2.begin = self._on_agent_collision

    def _on_agent_collision(self, arbiter, space, data):
        """Collision callback for agent-obstacle collisions"""
        self.collision_occurred = True
        return True  # Allow collision to proceed normally

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the environment

        Args:
            agent: Agent instance to add
        """
        # Remove existing agent if present
        if self.agent is not None:
            self.agent.unset_space()

        self.agent = agent
        # Set world size for the agent
        self.agent.world_size = self.world_size
        # Connect agent to this environment's space
        self.agent.set_space(self.space)

    def add_obstacle(self, obstacle: Entity) -> None:
        """
        Add an obstacle to the environment

        Args:
            obstacle: Entity instance to add as obstacle
        """
        if len(self.obstacles) >= self.max_obstacles:
            raise ValueError(
                f"Cannot add more than {self.max_obstacles} obstacles"
            )

        self.obstacles.append(obstacle)
        # Set world size for the obstacle
        obstacle.world_size = self.world_size
        # Set collision type to OBSTACLE for proper collision handling
        obstacle.collision_type = CollisionType.OBSTACLE
        obstacle.shape.collision_type = CollisionType.OBSTACLE
        # Connect obstacle to this environment's space
        obstacle.set_space(self.space)

    def clear_all_entities(self) -> None:
        """Clear all entities from the environment (public method)"""
        # Clear all entities
        for entity in self._get_all_entities():
            entity.unset_space()

        # Reset references
        self.agent = None
        self.obstacles = []

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        # Reset collision flag
        self.collision_occurred = False

        # Reset entities with random positions, ensuring no overlaps
        self._reset_entities_safely()

        self.target_position = self._get_random_position()

        observation = self._get_obs_dict()
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

        if self.agent is not None:
            self.agent.apply_action(action)

        # Update physics (collision handlers will set collision_occurred flag)
        dt = 1.0 / self.metadata["render_fps"]
        self.space.step(dt)

        # Update all entities (this will calculate their accelerations)
        for entity in self._get_all_entities():
            entity.update(dt)

        observation = self._get_obs_dict()
        reward = self._calculate_reward()
        terminated = self.collision_occurred
        truncated = self._check_out_of_bounds()

        return observation, reward, terminated, truncated, {}

    def render(self) -> None:
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))

        # Convert world coordinates to screen coordinates
        scale = self.screen_size / self.world_size
        offset = self.screen_size / 2

        # Draw target
        target_screen_x = int(self.target_position.x * scale + offset)
        target_screen_y = int(self.target_position.y * scale + offset)
        pygame.draw.circle(
            self.screen, (0, 255, 0), (target_screen_x, target_screen_y), 20, 3
        )

        # Draw entities using their render methods
        for entity in self._get_all_entities():
            entity.render(self.screen, scale, offset)

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

    def _get_random_position(self) -> Vector2D:
        half_size = self.world_size / 2 - 2  # Leave margin from walls
        x = random.uniform(-half_size, half_size)
        y = random.uniform(-half_size, half_size)
        return Vector2D(x, y)

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Return observation as dictionary with fixed-size numpy arrays"""
        # Agent observation: [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
        agent_obs = (
            self._encode_entity(self.agent)
            if self.agent
            else np.zeros(7, dtype=np.float32)
        )

        # Target observation: [pos_x, pos_y]
        target_obs = np.array(
            [self.target_position.x, self.target_position.y], dtype=np.float32
        )

        # Obstacles observation: (max_obstacles, 7) with padding
        obstacles_obs = np.zeros((self.max_obstacles, 7), dtype=np.float32)
        mask = np.zeros(self.max_obstacles, dtype=np.float32)

        for i, obstacle in enumerate(self.obstacles):
            if i >= self.max_obstacles:
                break

            obstacles_obs[i] = self._encode_entity(obstacle)
            mask[i] = 1.0  # Mark as valid obstacle

        return {
            "agent": agent_obs,
            "obstacles": obstacles_obs,
            "target": target_obs,
            "mask": mask,
        }

    def _encode_entity(self, entity: Entity) -> np.ndarray:
        """Encode an entity into 7D array: [radius, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]"""
        pos = entity.get_position()
        vel = entity.get_velocity()
        acc = entity.get_acceleration()

        return np.array(
            [
                entity.radius,
                pos.x,
                pos.y,
                vel.x,
                vel.y,
                acc.x,
                acc.y,
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self) -> float:
        if not self.agent:
            return 0.0

        agent_pos = self.agent.get_position()
        distance_to_target = np.sqrt(
            (agent_pos.x - self.target_position.x) ** 2
            + (agent_pos.y - self.target_position.y) ** 2
        )

        # Negative distance as reward (closer is better)
        reward = -distance_to_target * 0.1

        # Bonus for reaching target
        if distance_to_target < 1.0:
            reward += 10.0

        return reward

    def _check_out_of_bounds(self) -> bool:
        """Check if agent is out of world bounds"""
        if not self.agent:
            return False

        agent_pos = self.agent.get_position()
        half_size = self.world_size / 2

        # Check if agent center is outside bounds
        return abs(agent_pos.x) > half_size or abs(agent_pos.y) > half_size

    def _reset_entities_safely(self) -> None:
        """Reset all entities with random positions, ensuring no overlaps"""
        occupied_positions = []

        # Reset all entities with overlap checking (agent first, then obstacles)
        for entity in self._get_all_entities():
            max_attempts = 100
            for _ in range(max_attempts):
                # Use entity's own reset function
                entity.reset(self.world_size)
                entity_pos = entity.get_position()

                # Check if position is valid (not overlapping with anything)
                if self._is_position_safe(
                    entity_pos, entity.radius, occupied_positions
                ):
                    occupied_positions.append((entity_pos, entity.radius))
                    break

    def _is_position_safe(
        self,
        position: Vector2D,
        entity_radius: float,
        occupied_positions: List[Tuple[Vector2D, float]],
    ) -> bool:
        """Check if a position is safe (doesn't overlap with existing entities)"""
        for pos, radius in occupied_positions:
            distance = np.sqrt(
                (position.x - pos.x) ** 2 + (position.y - pos.y) ** 2
            )
            min_distance = entity_radius + radius + 0.5  # Add safety margin
            if distance < min_distance:
                return False
        return True

    def close(self) -> None:
        pygame.quit()
