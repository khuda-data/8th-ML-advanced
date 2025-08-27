import gymnasium as gym
from gymnasium import spaces
import pygame
from pygame import Vector2
import pymunk
import numpy as np
import math
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
        recognition_radius: float = 7.0,
        destruction_radius: float = 15.0,
        max_velocity: float = 10.0,
        max_acceleration: float = 5.0,
    ) -> None:
        super().__init__()

        self.elapsed_steps = 0

        self.render_mode = render_mode
        self.max_obstacles = max_obstacles
        self.target_radius = target_radius
        self.recognition_radius = recognition_radius
        self.destruction_radius = destruction_radius
        self.pre_distance_to_target = destruction_radius

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.position_scale = recognition_radius

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,  # radius, vel_x, vel_y, acc_x, acc_y
                ),
                "obstacles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.max_obstacles,
                        7,
                    ),  # radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y
                    dtype=np.float32,
                ),
                "target": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,  # rel_pos_x, rel_pos_y
                ),
                "mask": spaces.Box(
                    low=0, high=1, shape=(self.max_obstacles,), dtype=np.float32
                ),
            }
        )

        if self.render_mode == "human":
            pygame.init()
            self.screen_size = 800
            self.screen = pygame.display.set_mode(
                (self.screen_size, self.screen_size)
            )
            self.clock = pygame.time.Clock()
        elif self.render_mode == "rgb_array":
            pygame.init()
            self.screen_size = 800
            self.screen = pygame.Surface((self.screen_size, self.screen_size))
            self.clock = None
        else:
            self.screen_size = 800
            self.screen = None
            self.clock = None

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
        return False

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

        center_pos = self.agent.get_position() if self.agent else Vector2(0, 0)
        safe_position = self._find_safe_position_in(
            entity.radius,
            unsafe_areas,
            area_radius=self.destruction_radius,
            area_center=center_pos,
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
        self.pre_distance_to_target = self.destruction_radius
        self.elapsed_steps = 0

        if self.agent is None:
            self.add_agent()
        self._reset_entity(self.agent, Vector2(0, 0))
        agent_pos = self.agent.get_position()

        unsafe_areas = [(agent_pos, self.agent.radius)]
        for obstacle in self.obstacles:
            safe_position = self._find_safe_position_in(
                obstacle.radius,
                unsafe_areas,
                area_radius=self.destruction_radius,
                area_center=agent_pos,
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
        reward = self._calculate_reward(action)
        terminated = self.collision_occurred or self._check_target_reached()
        truncated = (
            self._is_out_destruction_(self.target_position)
            or self.elapsed_steps > 500
        )

        self.pre_distance_to_target = self._get_distance_to_target()

        return observation, reward, terminated, truncated, {}

    def _manage_obstacles(self):
        if not self.agent:
            return

        obstacles_to_adjust: List[Entity] = []
        for obstacle in self.obstacles:
            if self._is_out_destruction_(obstacle.get_position()):
                obstacles_to_adjust.append(obstacle)

        for obstacle in obstacles_to_adjust:
            unsafe_areas = [
                (e.get_position(), e.radius) for e in self._get_entities()
            ]

            pos = self._find_safe_position_in_ring(
                obstacle.radius,
                unsafe_areas,
                center=self.agent.get_position(),
                inner_radius=self.recognition_radius,
                outer_radius=self.destruction_radius,
            )
            obstacle.set_position(pos)
            obstacle.reset()

    def render(self):
        if self.render_mode is None or self.screen is None:
            return

        self.screen.fill((255, 255, 255))
        scale = self.screen_size / (2 * self.destruction_radius)

        agent_pos = self.agent.get_position() if self.agent else Vector2(0, 0)
        offset = (
            Vector2(self.screen_size / 2, self.screen_size / 2)
            - agent_pos * scale
        )
        screen_center = (self.screen_size // 2, self.screen_size // 2)

        destruction_screen_radius = int(self.screen_size / 2)
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            screen_center,
            destruction_screen_radius,
            2,
        )

        recognition_screen_radius = int(self.recognition_radius * scale)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            screen_center,
            recognition_screen_radius,
            2,
        )

        target_screen_pos = self.target_position * scale + offset
        target_screen_radius = int(self.target_radius * scale)

        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            (int(target_screen_pos.x), int(target_screen_pos.y)),
            target_screen_radius,
            3,
        )

        for entity in self._get_entities():
            entity.render(self.screen, scale, offset)

        if self.render_mode == "human":
            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            rgb_array = pygame.surfarray.array3d(self.screen)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))
            return rgb_array

    def _get_distance_to_target(self) -> float:
        return self.target_position.distance_to(self.agent.get_position())

    def _get_random_position(
        self, area_radius: float, center: Vector2 = Vector2(0, 0)
    ) -> Vector2:
        r = np.sqrt(self.np_random.uniform(0, (area_radius - 1.0) ** 2))
        theta = self.np_random.uniform(0, 2 * np.pi)
        x = center.x + r * np.cos(theta)
        y = center.y + r * np.sin(theta)
        return Vector2(x, y)

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        # Agent observation: radius, vel_x, vel_y, acc_x, acc_y (scaled)
        agent_obs = (
            self._encode_agent(self.agent)
            if self.agent
            else np.zeros(5, dtype=np.float32)
        )

        # Target observation: rel_pos_x, rel_pos_y (scaled)
        target_obs = (
            self._encode_target_relative(self.agent, self.target_position)
            if self.agent
            else np.zeros(2, dtype=np.float32)
        )

        # Obstacles observation: radius, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y, acc_x, acc_y (scaled)
        obstacles_obs = np.zeros((self.max_obstacles, 7), dtype=np.float32)
        mask = np.zeros(self.max_obstacles, dtype=np.float32)

        if self.agent:
            obs_idx = 0
            for obstacle in self.obstacles:
                if obs_idx >= self.max_obstacles:
                    break

                if self._is_in_recognition_(obstacle.get_position()):
                    obstacles_obs[obs_idx] = self._encode_obstacle_relative(
                        self.agent, obstacle
                    )
                    mask[obs_idx] = 1.0
                    obs_idx += 1

        return {
            "agent": agent_obs,
            "obstacles": obstacles_obs,
            "target": target_obs,
            "mask": mask,
        }

    def _encode_agent(self, agent: Agent) -> np.ndarray:
        vel = agent.get_velocity()
        acc = agent.get_acceleration()

        vel_scaled = Vector2(
            vel.x / self.max_velocity, vel.y / self.max_velocity
        )
        acc_scaled = Vector2(
            acc.x / self.max_acceleration, acc.y / self.max_acceleration
        )

        return np.array(
            [
                agent.radius,
                vel_scaled.x,
                vel_scaled.y,
                acc_scaled.x,
                acc_scaled.y,
            ],
            dtype=np.float32,
        )

    def _encode_target_relative(
        self, agent: Agent, target_pos: Vector2
    ) -> np.ndarray:
        agent_pos = agent.get_position()
        rel_pos = target_pos - agent_pos

        rel_pos_scaled = Vector2(
            rel_pos.x / self.position_scale, rel_pos.y / self.position_scale
        )

        return np.array([rel_pos_scaled.x, rel_pos_scaled.y], dtype=np.float32)

    def _encode_obstacle_relative(
        self, agent: Agent, obstacle: Entity
    ) -> np.ndarray:
        agent_pos = agent.get_position()
        agent_vel = agent.get_velocity()

        obs_pos = obstacle.get_position()
        obs_vel = obstacle.get_velocity()
        obs_acc = obstacle.get_acceleration()

        rel_pos = obs_pos - agent_pos
        rel_vel = obs_vel - agent_vel

        rel_pos_scaled = Vector2(
            rel_pos.x / self.position_scale, rel_pos.y / self.position_scale
        )

        rel_vel_scaled = Vector2(
            rel_vel.x / self.max_velocity, rel_vel.y / self.max_velocity
        )

        acc_scaled = Vector2(
            obs_acc.x / self.max_acceleration, obs_acc.y / self.max_acceleration
        )

        return np.array(
            [
                obstacle.radius,
                rel_pos_scaled.x,
                rel_pos_scaled.y,
                rel_vel_scaled.x,
                rel_vel_scaled.y,
                acc_scaled.x,
                acc_scaled.y,
            ],
            dtype=np.float32,
        )

    def _calculate_reward(self, action: np.ndarray) -> float:
        if not self.agent:
            return 0.0

        agent_pos = self.agent.get_position()

        target_distance = agent_pos.distance_to(self.target_position)
        max_distance = self.destruction_radius

        normalized_target_dist = max(1e-3, min(target_distance / max_distance, 1.0))
        sigmoid_target_dist = 1/(1+math.exp(normalized_target_dist-1))
        action_target_cosine_similarity = self._calculate_direction_cosine_similarity(self.target_position, self.agent.get_velocity())
        velovity_alpha = self.agent.get_velocity().length() / self.agent.max_velocity

        target_reward = velovity_alpha * action_target_cosine_similarity * sigmoid_target_dist * RewardType.TARGET_REWARD_SCALE 
            # target_reward -= RewardType.STOP_PENALTY

        # print("time_penalty", time_penalty)

        obstacle_penalty = 0.0
        obstacles_danger = 0.0

        for obstacle in self.obstacles:
            obstacle_pos = obstacle.get_position()
            obstacle_distance = agent_pos.distance_to(obstacle_pos)

            if obstacle_distance <= self.recognition_radius:
                if obstacle_distance < RewardType.SAFETY_RADIUS:
                    obstacles_danger += 1
                    action_obstacle_cosine_similarity = min(
                        -self._calculate_direction_cosine_similarity(
                                obstacle_pos, 
                                self.agent.get_velocity()
                            ), 
                    -1e-2)
                    normalized_obstacle_dist = max(1e-3, (RewardType.SAFETY_RADIUS - obstacle_distance) / RewardType.SAFETY_RADIUS)

                    penalty = -1 * normalized_obstacle_dist * action_obstacle_cosine_similarity * RewardType.OBSTACLE_PENALTY_SCALE
                    # print("p", penalty) 
                    # print("normalized_obstacle_dist",  math.log(normalized_obstacle_dist))

                    obstacle_penalty += 1e-2 if penalty < 0 else penalty

        # print("obstacle_penalty", obstacle_penalty)

        if obstacles_danger > 0:
            obstacle_penalty = obstacle_penalty / obstacles_danger

        target_bonus = 0.0
        if self._check_target_reached():
            target_bonus = RewardType.TARGET_BONUS

        collision_penalty = 0.0
        if self.collision_occurred:
            collision_penalty = RewardType.COLLISION_PENALTY

        boundary_penalty = 0.0
        if self._is_out_destruction_(self.target_position):
            boundary_penalty = RewardType.BOUNDARY_PENALTY

        # print("obstacle_penalty", obstacle_penalty)

        # print("target_reward", )

        time_penalty = self.elapsed_steps/500
        time_target_penalty_alpha = 1 - time_penalty

        total_reward = (
            target_reward
            - obstacle_penalty
            + target_bonus
            - collision_penalty
            - boundary_penalty
        ) * time_target_penalty_alpha

        if math.sqrt(action[0]**2 + action[1]**2) < 0.3:
            total_reward *= RewardType.STOP_PENALTY
            time_penalty *= RewardType.STOP_PENALTY
        else:
            time_penalty *= RewardType.TIME_PENALTY

        total_reward -= time_penalty

        return total_reward
    
    def _calculate_direction_cosine_similarity(
        self, entitiy_vectoer: Vector2, velocity: Vector2
    ) -> float:
        vector_to_entity = entitiy_vectoer - self.agent.get_position()
        
        if velocity.length_squared() < 1e-9 or vector_to_entity.length_squared() < 1e-9:
            # velocity가 없고 타겟에 도달한 경우 -> 1.0 (최대 유사도)
            if vector_to_entity.length_squared() < 1e-9:
                return 1.0
            # 그 외 방향성 없는 경우는 0.0
            return 0.0

        normalized_velocity_vector = velocity.normalize()
        normalized_direction_to_entitiy = vector_to_entity.normalize()

        cosine_similarity = normalized_velocity_vector.dot(normalized_direction_to_entitiy)
    
        return np.clip(cosine_similarity, -1.0, 1.0)

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

    def _find_safe_position_in(
        self,
        position_radius: float,
        unsafe_areas: List[Tuple[Vector2, float]],
        area_radius: float,
        area_center: Vector2,
        max_attempts: int = 100,
    ) -> Vector2:
        for _ in range(max_attempts):
            position = self._get_random_position(
                area_radius=area_radius, center=area_center
            )
            if self._is_safe_area((position, position_radius), unsafe_areas):
                return position
        return area_center

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
