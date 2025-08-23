import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame import Vector2
import pymunk
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Type

from .types import CollisionType, RewardType
from .entities import Agent, Entity, StableObstacle


class KFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_obstacles: int = 10,
        target_radius: float = 1.0,
        recognition_radius: float = 7.0,  # 인식 범위
        destruction_radius: float = 15.0,  # 파괴 거리
    ) -> None:
        super().__init__()

        self.elapsed_steps = 0

        self.render_mode = render_mode
        self.max_obstacles = max_obstacles
        self.target_radius = target_radius
        self.recognition_radius = recognition_radius
        self.destruction_radius = destruction_radius

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

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
        self.target_position: Vector2 = Vector2(0, 0)

        self.collision_occurred = False

        self.space.on_collision(
            CollisionType.AGENT,
            CollisionType.OBSTACLE,
            begin=self._on_agent_collision,
        )
        self.space.on_collision(
            CollisionType.AGENT,
            CollisionType.ENTITY,
            begin=self._on_agent_collision,
        )

        self.space.on_collision(
            CollisionType.OBSTACLE,
            CollisionType.OBSTACLE,
        )

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
        self, obstacle_class: Type[Entity] = StableObstacle, **kwargs
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

        unsafe_areas = [
            (e.get_position(), e.radius) for e in self._get_entities()
        ]

        # 에이전트 위치를 중심으로 안전한 위치 탐색
        center_pos = self.agent.get_position() if self.agent else Vector2(0, 0)
        safe_position = self._find_safe_position(
            entity.radius,
            unsafe_areas,
            area_radius=self.destruction_radius,
            center=center_pos,
        )
        self._reset_entity(entity, safe_position)
        entity.set_space(self.space)
        return entity

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
        self.elapsed_steps = 0

        if self.agent is None:
            self.add_agent()
        self._reset_entity(self.agent, Vector2(0, 0))
        agent_pos = self.agent.get_position()

        unsafe_areas = [(agent_pos, self.agent.radius)]
        for obstacle in self.obstacles:
            safe_position = self._find_safe_position(
                obstacle.radius,
                unsafe_areas,
                area_radius=self.destruction_radius,
                center=agent_pos,
            )
            self._reset_entity(obstacle, safe_position)
            unsafe_areas.append((safe_position, obstacle.radius))

        self.target_position = self._get_random_position(
            area_radius=self.destruction_radius, center=agent_pos
        )

        observation = self._get_obs_dict()
        return observation, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        if self.agent is not None:
            self.agent.apply_action(action)

        self.elapsed_steps += 1
        dt = 1.0 / self.metadata["render_fps"]
        self.space.step(dt)

        for entity in self._get_entities():
            entity.update(dt)

        self._manage_obstacles()

        observation = self._get_obs_dict()
        reward = self._calculate_reward()
        terminated = self.collision_occurred or self._check_target_reached()
        truncated = self._is_out_destruction_(self.target_position)

        return observation, reward, terminated, truncated, {}

    def _manage_obstacles(self):
        if not self.agent:
            return

        obstacles_to_remove = []
        for obstacle in self.obstacles:
            if self._is_out_destruction_(obstacle.get_position()):
                obstacles_to_remove.append(obstacle)

        for obstacle in obstacles_to_remove:
            obstacle.unset_space()
            self.obstacles.remove(obstacle)

            new_obstacle = StableObstacle(
                radius=obstacle.radius, mass=obstacle.mass, speed=obstacle.speed
            )
            unsafe_areas = [
                (e.get_position(), e.radius) for e in self._get_entities()
            ]

            pos = self._find_safe_position_in_ring(
                new_obstacle.radius,
                unsafe_areas,
                center=self.agent.get_position(),
                inner_radius=self.recognition_radius,
                outer_radius=self.destruction_radius,
            )
            self._reset_entity(new_obstacle, pos)
            new_obstacle.set_space(self.space)
            self.obstacles.append(new_obstacle)

    def render(self) -> None:
        if self.render_mode is None:
            return

        self.screen.fill((255, 255, 255))
        scale = self.screen_size / (2 * self.destruction_radius)

        agent_pos = self.agent.get_position() if self.agent else Vector2(0, 0)
        offset = (
            Vector2(self.screen_size / 2, self.screen_size / 2)
            - agent_pos * scale
        )
        screen_center = (self.screen_size // 2, self.screen_size // 2)

        # 파괴 범위 링 그리기 (빨간색)
        destruction_screen_radius = int(self.screen_size / 2)
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),  # 빨간색
            screen_center,
            destruction_screen_radius,
            2,
        )

        # 인식 범위 링 그리기 (초록색)
        recognition_screen_radius = int(self.recognition_radius * scale)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),  # 초록색
            screen_center,
            recognition_screen_radius,
            2,
        )

        target_screen_pos = self.target_position * scale + offset
        target_screen_radius = int(self.target_radius * scale)

        # if self._is_in_recognition_(self.target_position):
        #     pygame.draw.circle(
        #         self.screen,
        #         (0, 255, 0),
        #         (int(target_screen_pos.x), int(target_screen_pos.y)),
        #         target_screen_radius,
        #         3,
        #     )
        # else:
        #     pygame.draw.circle(
        #         self.screen,
        #         (120, 120, 120),
        #         (int(target_screen_pos.x), int(target_screen_pos.y)),
        #         target_screen_radius,
        #         3,
        #     )

        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (int(target_screen_pos.x), int(target_screen_pos.y)),
            target_screen_radius,
            3,
        )

        for entity in self._get_entities():
            if self._is_in_recognition_(entity.get_position()):
                # if agent_pos.distance_to(entity.get_position()) > 0:
                #     entity.color = (255, 100, 100)
                entity.render(self.screen, scale, offset)
            # else:
            #     entity.color = (120, 120, 120)
            #     entity.render(self.screen, scale, offset)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def _get_random_position(
        self, area_radius: float, center: Vector2 = Vector2(0, 0)
    ) -> Vector2:
        r = np.sqrt(self.np_random.uniform(0, (area_radius - 1.0) ** 2))
        theta = self.np_random.uniform(0, 2 * np.pi)
        x = center.x + r * np.cos(theta)
        y = center.y + r * np.sin(theta)
        return Vector2(x, y)

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        agent_obs = (
            self._encode_entity(self.agent)
            if self.agent
            else np.zeros(7, dtype=np.float32)
        )
        target_obs = np.array(
            [self.target_position.x, self.target_position.y], dtype=np.float32
        )
        obstacles_obs = np.zeros((self.max_obstacles, 7), dtype=np.float32)
        mask = np.zeros(self.max_obstacles, dtype=np.float32)

        if self.agent:
            obs_idx = 0
            for obstacle in self.obstacles:
                # 관측 가능한 최대 장애물 수를 넘으면 중단
                if obs_idx >= self.max_obstacles:
                    break

                # 에이전트와의 거리를 계산하여 recognition_radius 안에 있을 때만 관측에 포함
                if self._is_in_recognition_(obstacle.get_position()):
                    obstacles_obs[obs_idx] = self._encode_entity(obstacle)
                    mask[obs_idx] = 1.0
                    obs_idx += 1

        # for i, obstacle in enumerate(self.obstacles):
        #     if i >= self.max_obstacles:
        #         break

        #     obstacles_obs[i] = self._encode_entity(obstacle)
        #     mask[i] = 1.0  # Mark as valid obstacle

        return {
            "agent": agent_obs,
            "obstacles": obstacles_obs,
            "target": target_obs,
            "mask": mask,
        }

    def _encode_entity(self, entity: Entity) -> np.ndarray:
        pos = entity.get_position()
        vel = entity.get_velocity()
        acc = entity.get_acceleration()
        return np.array(
            [entity.radius, pos.x, pos.y, vel.x, vel.y, acc.x, acc.y],
            dtype=np.float32,
        )

    def _calculate_reward(self) -> float:
        if not self.agent:
            return 0.0

        agent_pos = self.agent.get_position()
        distance_to_target = agent_pos.distance_to(self.target_position)
        reward = -distance_to_target * RewardType.DISTANCE_ALPHA

        if self._check_target_reached():
            reward += RewardType.TARGET_REACHED
        elif self.collision_occurred:
            reward -= RewardType.COLLISION_OCCURRED
        elif self._is_out_destruction_(self.target_position):
            reward -= RewardType.TARGET_DESTROYED

        reward -= self._get_time_penalty()
        return reward

    def _get_time_penalty(self) -> float:
        dt = 1.0 / self.metadata["render_fps"]
        return RewardType.TIME_ALPHA * self.elapsed_steps * dt

    def _check_target_reached(self) -> bool:
        if not self.agent:
            return False
        agent_pos = self.agent.get_position()
        return agent_pos.distance_to(self.target_position) < self.target_radius

    def _is_out_destruction_(self, vector: Vector2):
        agent_pos = self.agent.get_position()

        return agent_pos.distance_to(vector) > self.destruction_radius

    def _is_in_recognition_(self, vector: Vector2):
        agent_pos = self.agent.get_position()

        return agent_pos.distance_to(vector) < self.recognition_radius

    def _reset_entity(self, entity: Entity, position: Vector2):
        entity.set_position(position)
        entity.reset()

    def _find_safe_position(
        self,
        radius: float,
        unsafe_areas: List[Tuple[Vector2, float]],
        area_radius: float,
        center: Vector2,
        max_attempts: int = 100,
    ) -> Vector2:
        for _ in range(max_attempts):
            position = self._get_random_position(
                area_radius=area_radius, center=center
            )
            if self._is_safe_area((position, radius), unsafe_areas):
                return position
        return center

    def _find_safe_position_in_ring(
        self,
        radius: float,
        unsafe_areas: List[Tuple[Vector2, float]],
        center: Vector2,
        inner_radius: float,
        outer_radius: float,
        max_attempts: int = 100,
    ) -> Vector2:
        for _ in range(max_attempts):
            r = np.sqrt(
                self.np_random.uniform(inner_radius**2, outer_radius**2)
            )
            theta = self.np_random.uniform(0, 2 * np.pi)
            position = Vector2(
                center.x + r * np.cos(theta), center.y + r * np.sin(theta)
            )

            if self._is_safe_area((position, radius), unsafe_areas):
                return position
        return Vector2(center.x + outer_radius, center.y)

    def _is_safe_area(
        self,
        candidate_area: Tuple[Vector2, float],
        unsafe_areas: List[Tuple[Vector2, float]],
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
