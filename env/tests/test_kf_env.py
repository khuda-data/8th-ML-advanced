# tests/test_kf_env.py
import os
import math
import numpy as np
import pytest
from pygame import Vector2

# 창 없는 환경에서 pygame이 윈도우를 띄우지 않도록 설정
@pytest.fixture(scope="session", autouse=True)
def _headless_pygame():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# 실제 모듈 import (프로젝트 루트에서 pytest 실행 가정)
from env.kf_env import KFEnv


@pytest.fixture
def env():
    e = KFEnv(render_mode=None)
    try:
        yield e
    finally:
        # 안전하게 종료
        e.close()


def test_init_default_values():
    env = KFEnv()
    try:
        # 기본값(현재 구현 기준)
        assert env.render_mode is None
        assert env.max_obstacles == 10
        assert env.target_radius == 1.0
        # recognition / destruction 기본값 (kf_env.py 구현에 따름)
        assert env.recognition_radius == 7.0
        assert env.destruction_radius == 15.0

        # action space
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)

        # observation space keys and shapes/dtypes
        obs_space = env.observation_space
        assert "agent" in obs_space.spaces
        assert "obstacles" in obs_space.spaces
        assert "target" in obs_space.spaces
        assert "mask" in obs_space.spaces

        # shapes/dtypes as defined in constructor
        agent_space = obs_space.spaces["agent"]
        obstacles_space = obs_space.spaces["obstacles"]
        target_space = obs_space.spaces["target"]
        mask_space = obs_space.spaces["mask"]

        assert agent_space.shape == (7,)
        assert agent_space.dtype == np.float32

        assert obstacles_space.shape == (env.max_obstacles, 7)
        assert obstacles_space.dtype == np.float32

        assert target_space.shape == (2,)
        assert target_space.dtype == np.float32

        assert mask_space.shape == (env.max_obstacles,)
        assert mask_space.dtype == np.float32
    finally:
        env.close()


def test_init_custom_values():
    env = KFEnv(render_mode="human", max_obstacles=5, target_radius=2.0, recognition_radius=8.0, destruction_radius=20.0)
    try:
        assert env.render_mode == "human"
        assert env.max_obstacles == 5
        assert env.target_radius == 2.0
        assert env.recognition_radius == 8.0
        assert env.destruction_radius == 20.0
    finally:
        env.close()


def test_reset_observation_structure_and_dtypes(env):
    obs, info = env.reset(seed=0)

    assert isinstance(obs, dict)
    assert isinstance(info, dict)

    assert obs["agent"].shape == (7,)
    assert obs["agent"].dtype == np.float32

    assert obs["obstacles"].shape == (env.max_obstacles, 7)
    assert obs["obstacles"].dtype == np.float32

    assert obs["target"].shape == (2,)
    assert obs["target"].dtype == np.float32

    assert obs["mask"].shape == (env.max_obstacles,)
    assert obs["mask"].dtype == np.float32


def test_step_contract_and_elapsed_steps(env):
    env.reset(seed=1)
    before = env.elapsed_steps
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.elapsed_steps == before + 1


def test_step_extreme_actions_do_not_crash(env):
    env.reset()
    # extreme values within action_space bounds
    for a in [np.array([1.0, 1.0], dtype=np.float32), np.array([-1.0, -1.0], dtype=np.float32)]:
        obs, reward, terminated, truncated, info = env.step(a)
        # 기본적인 반환형 검증
        assert isinstance(obs, dict)
        assert isinstance(reward, float)


def test_reward_is_finite(env):
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert not np.isinf(reward)


def test_termination_on_target_reached(env):
    env.reset()
    # 에이전트를 타깃 위치로 강제로 이동 -> terminated True
    env.agent.set_position(Vector2(env.target_position.x, env.target_position.y))
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_termination_on_collision_flag(env):
    env.reset()
    env.collision_occurred = True
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    assert truncated is False


def test_truncation_on_step_limit(env):
    env.reset()
    env.elapsed_steps = 5000
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert truncated is True


def test_observation_bounds_within_destruction_radius(env):
    obs, _ = env.reset(seed=2)
    # agent 위치는 reset에서 (0,0)으로 세팅되므로 target만 검사
    target = obs["target"]
    # target은 agent 중심으로 destruction_radius 내에 생성됨
    assert -env.destruction_radius - 1e-6 <= float(target[0]) <= env.destruction_radius + 1e-6
    assert -env.destruction_radius - 1e-6 <= float(target[1]) <= env.destruction_radius + 1e-6


def test_multiple_resets_change_target(env):
    # agent는 매 reset에서 동일(0,0)으로 초기화되므로 target 변화 여부를 확인
    targets = []
    for s in [0, 1, 2]:
        _, _ = env.reset(seed=s)
        targets.append((float(env.target_position.x), float(env.target_position.y)))
    # 서로 다른 시드로 reset하면 적어도 하나는 달라야 함
    assert len(set(targets)) > 1


def test_obstacle_mask_consistency(env):
    env = KFEnv(max_obstacles=5)
    try:
        obs, _ = env.reset()
        mask = obs["mask"]
        obstacles = obs["obstacles"]
        for i in range(env.max_obstacles):
            if mask[i] == 0:
                # mask가 0인 행은 모두 0으로 채워져 있어야 함
                assert np.allclose(obstacles[i], 0.0)
    finally:
        env.close()


def test_add_obstacle_capacity_limit():
    env = KFEnv(max_obstacles=1)
    try:
        env.add_obstacle()
        with pytest.raises(ValueError):
            env.add_obstacle()
    finally:
        env.close()


def test_manage_obstacles_relocates_outside(env):
    env.reset(seed=0)
    ob = env.add_obstacle()
    agent_pos = env.agent.get_position()
    # 파괴반경 밖으로 강제 이동
    far_pos = Vector2(agent_pos.x + env.destruction_radius + 5.0, agent_pos.y)
    ob.set_position(far_pos)
    ob.reset()

    env._manage_obstacles()
    new_pos = ob.get_position()
    dist = new_pos.distance_to(agent_pos)
    assert dist >= env.recognition_radius - 1e-6
    assert dist <= env.destruction_radius + 1e-6


def test_close_and_render_behaviour():
    env = KFEnv(render_mode=None)
    try:
        env.reset()
        # render()는 render_mode가 None이면 None 반환
        assert env.render() is None
        # close() 호출 시 예외 발생하지 않음
        env.close()
    except Exception:
        # ensure close doesn't raise
        env.close()
        raise
