from env.kf_env import KFEnv
from sac.kf_policy import KFSACPolicy
from stable_baselines3 import SAC
import torch.nn as nn

env = KFEnv()

model = SAC(
    policy=KFSACPolicy,
    env=env,
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    learning_starts=5_000,
    ent_coef="auto",
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=nn.ReLU,
        log_std_init=-3.0,
    ),
    device="auto",
    verbose=1,
)

model.learn(total_timesteps=300_000)

from env.kf_env import KFEnv
from sac.kf_policy import KFSACPolicy
from stable_baselines3 import SAC
import torch.nn as nn

env = KFEnv()

model = SAC(
    policy=KFSACPolicy,    # 사용할 정책 클래스 
    env=env,               # 학습할 환경 

    # 학습 관련 하이퍼파라미터
    learning_rate=3e-4,    # Adam 옵티마이저 학습률
    buffer_size=1_000_000, # 리플레이 버퍼 최대 크기 (저장 가능한 transition 개수)
    batch_size=256,        # 한 번 학습할 때 샘플링할 미니배치 크기
    tau=0.005,             # 타깃 네트워크 soft update 계수 (작을수록 안정적, 클수록 빠르게 업데이트)
    gamma=0.99,            # 할인율 (미래 보상의 현재 가치 반영 정도)
    learning_starts=5_000, # 학습 시작 전 리플레이 버퍼에 쌓아둘 transition 개수
    ent_coef="auto",       # 엔트로피 계수

    # 정책 네트워크(Actor & Critic) 구조
    policy_kwargs=dict(
        net_arch=[256, 256],   # 은닉층 크기 
        activation_fn=nn.ReLU, # 은닉층 활성화 함수
        log_std_init=-3.0,     # 액션 분포의 초기 log-std 값 (분산 크기 조절)
        # features_extractor_class=DNN_Extractor,            # 필요시 커스텀 특징 추출기 사용
        # features_extractor_kwargs=dict(features_dim=128),  # 특징 추출기 하이퍼파라미터
    ),

    # 실행 환경 설정
    device="auto",         # cuda cpu 자동 선택
    verbose=1,             # 로깅 수준 (0 = 최소, 1 = 정보, 2 = 디버그)
)

# 총 스텝
model.learn(total_timesteps=300_000)

# 학습된 모델 저장
model.save("sac")
