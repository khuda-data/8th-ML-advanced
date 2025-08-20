import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Type

from .types import CollisionType, Vector2D
from .entities import Agent, Entity


class KFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: float = 20.0,
        max_obstacles: int = 10,
        target_radius: float = 1.0,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.max_obstacles = max_obstacles
        self.target_radius = target_radius

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
        self.space.gravity = (0, 0)

        self.agent: Optional[Agent] = None
        self.obstacles: List[Entity] = []
        self.target_position: Vector2D = Vector2D(0, 0)

        self.collision_occurred = False

        handler = self.space.add_collision_handler(
            CollisionType.AGENT, CollisionType.OBSTACLE
        )
        handler.begin = self._on_agent_collision

        handler2 = self.space.add_collision_handler(
            CollisionType.AGENT, CollisionType.ENTITY
        )
        handler2.begin = self._on_agent_collision

        self.reset()

    def _get_entities(self) -> List[Entity]:
        entities = []
        if self.agent is not None:
            entities.append(self.agent)
        entities.extend(self.obstacles)
        return entities

    def _on_agent_collision(self, arbiter, space, data):
        self.collision_occurred = True
        return True

    def add_agent(self, agent_class: Type[Agent] = Agent, **kwargs) -> Agent:
        if self.agent is not None:
            self.agent.unset_space()

        self.agent = self._add_entity(agent_class, **kwargs)
        return self.agent

    def add_obstacle(
        self, obstacle_class: Type[Entity] = Entity, **kwargs
    ) -> Entity:
        if len(self.obstacles) >= self.max_obstacles:
            raise ValueError(
                f"Cannot add more than {self.max_obstacles} obstacles"
            )

        obstacle = self._add_entity(obstacle_class, **kwargs)
        self.obstacles.append(obstacle)
        return obstacle

    def _add_entity(
        self, entity_class: Type[Entity] = Entity, **kwargs
    ) -> Entity:
        entity = entity_class(**kwargs)

        unsafe_areas = []
        for other_entity in self._get_entities():
            if other_entity != entity:
                unsafe_areas.append(
                    (other_entity.get_position(), other_entity.radius)
                )

        self._reset_entity(
            entity, self._find_safe_position(entity.radius, unsafe_areas)
        )

        entity.set_space(self.space)

    def clear_entities(self) -> None:
        for entity in self._get_entities():
            entity.unset_space()

        self.agent = None
        self.obstacles = []

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)

        self.collision_occurred = False

        unsafe_areas = []

        for entity in self._get_entities():
            safe_position = self._find_safe_position(
                entity.radius, unsafe_areas
            )
            self._reset_entity(entity, safe_position)
            unsafe_areas.append((safe_position, entity.radius))

        self.target_position = self._get_random_position()

        observation = self._get_obs_dict()
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:

        if self.agent is not None:
            self.agent.apply_action(action)

        # Update physics
        dt = 1.0 / self.metadata["render_fps"]
        self.space.step(dt)

        for entity in self._get_entities():
            entity.update(dt)

        observation = self._get_obs_dict()
        reward = self._calculate_reward()
        terminated = self.collision_occurred or self._check_target_reached()
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
        target_screen_radius = int(self.target_radius * scale)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (target_screen_x, target_screen_y),
            target_screen_radius,
            3,
        )

        for entity in self._get_entities():
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

    def _get_random_position(self, radius: float = 2.0) -> Vector2D:
        half_size = self.world_size / 2 - radius
        x = self.np_random.uniform(-half_size, half_size)
        y = self.np_random.uniform(-half_size, half_size)
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
        distance_to_target = agent_pos.distance_to(self.target_position)

        reward = -distance_to_target * 0.1

        if self._check_target_reached():
            reward += 100.0

        return reward

    def _check_target_reached(self) -> bool:
        if not self.agent:
            return False

        agent_pos = self.agent.get_position()
        distance_to_target = agent_pos.distance_to(self.target_position)

        return distance_to_target < self.target_radius

    def _check_out_of_bounds(self) -> bool:
        """Check if agent is out of world bounds"""
        if not self.agent:
            return False

        agent_pos = self.agent.get_position()
        half_size = self.world_size / 2

        # Check if agent center is outside bounds
        return abs(agent_pos.x) > half_size or abs(agent_pos.y) > half_size

    def _reset_entity(self, entity: Entity, position: Vector2D):
        entity.set_position(position)
        entity.reset()

    def _find_safe_position(
        self,
        radius: float,
        unsafe_areas: List[Tuple[Vector2D, float]],
        max_attempts: int = 100,
    ) -> Optional[Vector2D]:
        for _ in range(max_attempts):
            position = self._get_random_position(radius=radius)

            if self._is_safe_area((position, radius), unsafe_areas):
                return position

        return Vector2D(0, 0)

    def _is_safe_area(
        self,
        candidate_area: Tuple[Vector2D, float],
        unsafe_areas: List[Tuple[Vector2D, float]],
        margin=0.5,
    ) -> bool:
        candidate_position, candidate_radius = candidate_area
        for unsafe_position, unsafe_radius in unsafe_areas:
            distance = candidate_position.distance_to(unsafe_position)
            min_distance = candidate_radius + unsafe_radius + margin
            if distance < min_distance:
                return False
        return True

    def close(self) -> None:
        pygame.quit()
