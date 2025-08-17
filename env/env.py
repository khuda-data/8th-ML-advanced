import gymnasium as gym
from gymnasium import spaces
import pygame
import pymunk
import numpy as np


class KFEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 2. Action과 Observation Space 정의
        # 예: Agent에 좌/우 힘을 가하는 액션
        self.action_space = spaces.Discrete(2)
        # 예: Agent의 위치와 속도를 관측
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

        # 3. Pygame 및 Pymunk 초기화 (렌더링에 필요)
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()

        # 4. 물리 엔진(Pymunk) 공간 생성
        self.space = pymunk.Space()
        self.space.gravity = (0, 900)  # 중력 설정

        # 여기에 Agent, 장애물 등 나만의 오브젝트를 생성하고 self.space에 추가
        # self.agent_body = pymunk.Body(...)
        # self.agent_shape = pymunk.Circle(...)
        # self.space.add(self.agent_body, self.agent_shape)
        # ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 모든 오브젝트의 위치, 속도 등을 초기 상태로 리셋
        # ...

        # 초기 관측값과 정보 반환
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # 1. Action을 물리 엔진에 적용 (예: Agent Body에 힘을 가함)
        # if action == 0: self.agent_body.apply_force_at_local_point(...)

        # 2. 물리 시뮬레이션을 한 스텝 진행
        dt = 1.0 / self.metadata["render_fps"]
        self.space.step(dt)

        # 3. 상태 업데이트 및 보상 계산
        # 충돌 감지: Pymunk의 collision handler를 사용하거나 직접 계산
        # if is_collided: reward = -10; terminated = True

        observation = self._get_obs()
        reward = ...
        terminated = ...  # 에피소드 종료 조건
        truncated = False  # 시간 초과 조건
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        # 4. Pygame을 사용해 현재 물리 상태를 화면에 그리기
        if self.render_mode is None:
            return

        self.screen.fill("white")  # 배경 채우기

        # self.space 안의 모든 오브젝트(shape)를 순회하며 그리기
        for shape in self.space.shapes:
            # ... Pygame의 그리기 함수(pygame.draw.circle 등) 사용 ...
            pass

        if self.render_mode == "human":
            pygame.display.flip()  # 화면 업데이트
            self.clock.tick(self.metadata["render_fps"])

    def _get_obs(self):
        # 현재 Agent의 상태를 Observation Space 형식에 맞게 반환
        return np.array([self.agent_body.position.x, ...])

    def _get_info(self):
        # 추가 정보 (디버깅용)
        return {}

    def close(self):
        # 5. 환경 종료 시 자원 해제
        pygame.quit()
