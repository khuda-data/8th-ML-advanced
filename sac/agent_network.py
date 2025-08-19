import torch
import torch.nn as nn
from torch.distributions.normal import Normal # 연속적인 행동을 위한 정규분포

# PaddingEncoder의 출력 크기와 동일하게 state_size를 설정
# (Agent: 4) + (Target: 2) + (Obstacles: 3 * 10) = 36
state_size = 36
# 출력: 나의 다음 목표 속도 벡터(vx, vy) (2)
action_size = 2

class ActorNetwork(nn.Module):
    """
    PaddingEncoder로부터 받은 상태(state)를 입력받아
    행동 확률 분포(정책)를 출력하는 액터(Actor) 신경망입니다.
    """
    def __init__(self, layer_size=128):
        """
        신경망의 레이어를 초기화합니다.
        """
        super(ActorNetwork, self).__init__()
        
        # --- 정책 네트워크 (Policy Network) ---
        self.policy_layers = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU()
        )
        
        # 행동의 평균(mean)을 결정하는 출력 레이어입니다.
        self.actor_mean = nn.Linear(layer_size, action_size)
        
        # 행동의 표준편차(log_std)를 결정하는 학습 가능한 파라미터입니다.
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_size))
        

    def forward(self, state):
        """
        신경망의 순전파(forward pass)를 정의합니다.
        """
        # 공통 레이어를 통과시켜 핵심 특징을 추출합니다.
        features = self.policy_layers(state)
        
        # 행동 분포의 평균을 계산합니다.
        action_mean = self.actor_mean(features)
        
        # 로그 표준편차로부터 실제 표준편차를 계산합니다.
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # 평균과 표준편차를 이용해 정규 분포(정책)를 생성합니다.
        policy_dist = Normal(action_mean, action_std)
        
        # 최종적으로 행동 분포를 반환합니다.
        return policy_dist