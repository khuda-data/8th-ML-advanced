# test_kf_env.py
import os
import math
import pytest
import numpy as np
from pygame import Vector2

# 창 없는 환경에서 pygame 초기화가 실패하지 않도록 설정
@pytest.fixture(scope="session", autouse=True)
def _headless_pygame():
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# KFEnv 가져오기
from env.kf_env import KFEnv


@pytest.fixture
def env():
    e = KFEnv(render_mode=None)
    try:
        yield e
    finally:
        e.close()


@pytest.mark.parametrize("max_obs", [1, 5, 10])
def test_api_contract_and_spaces(max_obs):
    env = KFEnv(render_mode=None, max_obstacles=max_obs)
    try:
        obs, info = env.reset(seed=123)

        # observation 스키마 및 dtype 확인
        assert set(obs.keys()) == {"agent", "obstacles", "target", "mask"}
        assert obs["agent"].shape == (7,) and obs["agent"].dtype == np.float32
        assert obs["obstacles"].shape == (max_obs, 7) and obs["obstacles"].dtype == np.float32
        assert obs["target"].shape == (2,) and obs["target"].dtype == np.float32
        assert obs["mask"].shape == (max_obs,) and obs["mask"].dtype == np.float32

        # action space 사양 확인
        assert env.action_space.shape == (2,)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
    finally:
        env.close()


def test_step_contract_and_elapsed_steps(env):
    env.reset(seed=0)
    before = env.elapsed_steps
    action = np.zeros(2, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    assert isinstance(obs, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert env.elapsed_steps == before + 1  # step 호출 시 증가


def test_termination_on_target_reached(env):
    env.reset(seed=0)
    # 에이전트를 타깃 위치로 이동시켜 즉시 종료 유도
    env.agent.set_position(Vector2(env.target_position.x, env.target_position.y))
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    # 타깃과의 거리가 0이므로 파괴반경 기반 truncated는 False
    assert truncated is False


def test_termination_on_collision_flag(env):
    env.reset(seed=0)
    env.collision_occurred = True
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated is True
    # 충돌만으로는 truncated 아님
    assert truncated is False


def test_truncation_on_step_limit(env):
    env.reset(seed=0)
    # 내부 로직: step 시작 시 +1 후 5000 초과면 truncated
    env.elapsed_steps = 5000
    _, _, terminated, truncated, _ = env.step(np.zeros(2, dtype=np.float32))
    assert truncated is True
    # 단순 스텝 제한으로는 terminated가 아닐 수 있음
    assert terminated in (False, True)


def test_add_obstacle_capacity_limit():
    env = KFEnv(render_mode=None, max_obstacles=1)
    try:
        env.add_obstacle()
        with pytest.raises(ValueError):
            env.add_obstacle()  # 정원 초과 시 예외
    finally:
        env.close()


def test_manage_obstacles_relocates_outside(env):
    # 파괴반경 밖의 장애물이 링 영역으로 재배치되는지 확인
    env.reset(seed=0)
    obs_before, _ = env.reset(seed=1)
    ob = env.add_obstacle()
    agent_pos = env.agent.get_position()
    # 파괴반경 바깥으로 밀어냄
    far_pos = Vector2(agent_pos.x + env.destruction_radius + 5.0, agent_pos.y)
    ob.set_position(far_pos)
    ob.reset()

    # 내부 재배치 로직 수행
    env._manage_obstacles()

    new_pos = ob.get_position()
    dist = new_pos.distance_to(agent_pos)
    assert env.recognition_radius <= dist <= env.destruction_radius + 1e-6


def test_mask_reflects_recognition_radius(env):
    env.reset(seed=0)
    # 관측 반경 내/외 한 개씩 배치
    o1 = env.add_obstacle()
    o2 = env.add_obstacle()
    agent_pos = env.agent.get_position()

    pos_in = Vector2(agent_pos.x + env.recognition_radius * 0.5, agent_pos.y)
    pos_out = Vector2(agent_pos.x + env.recognition_radius + 2.0, agent_pos.y)
    # 파괴반경 이내로 보장
    assert pos_out.distance_to(agent_pos) < env.destruction_radius - 1.0

    o1.set_position(pos_out); o1.reset()
    o2.set_position(pos_in);  o2.reset()

    obs = env._get_obs_dict()
    # 마스크는 관측 반경 내 장애물 수만큼 1이어야 함
    assert int(obs["mask"].sum()) == 1

    # 첫 번째 활성 장애물 행의 좌표가 pos_in과 일치하는지 확인
    row0 = obs["obstacles"][0]
    seen_pos = Vector2(float(row0[1]), float(row0[2]))
    assert math.isclose(seen_pos.x, pos_in.x, rel_tol=0, abs_tol=1e-3)
    assert math.isclose(seen_pos.y, pos_in.y, rel_tol=0, abs_tol=1e-3)


def test_find_safe_position_in_no_overlap(env):
    env.reset(seed=0)
    agent_pos = env.agent.get_position()
    unsafe = [(agent_pos, env.agent.radius)]
    # 후보 반경 1.0, 파괴반경 내에서 안전한 위치를 찾아야 함
    pos = env._find_safe_position_in(
        position_radius=1.0,
        unsafe_areas=unsafe,
        area_radius=env.destruction_radius,
        area_center=agent_pos,
        max_attempts=200,
    )
    min_dist = 1.0 + env.agent.radius + 0.5  # 구현상의 margin=0.5
    assert pos.distance_to(agent_pos) >= min_dist - 1e-6


def test_reward_monotonic_with_distance(env):
    # 거리 감소 시(기타 조건 동일) 보상이 증가해야 함
    env.reset(seed=0)

    agent_pos = env.agent.get_position()
    target = env.target_position

    # 멀리/가깝게 위치 지정
    dir_vec = target - agent_pos
    if dir_vec.length() == 0:
        dir_vec = Vector2(1.0, 0.0)
    unit = dir_vec.normalize()

    far = target + unit * (env.recognition_radius * 0.9)
    near = target + unit * (env.target_radius * 0.1)

    env.elapsed_steps = 0
    env.agent.set_position(far)
    r_far = env._calculate_reward()

    env.elapsed_steps = 0
    env.agent.set_position(near)
    r_near = env._calculate_reward()

    assert r_near > r_far  # 거리 항이 -alpha * d 이므로 가까울수록 보상 큼
