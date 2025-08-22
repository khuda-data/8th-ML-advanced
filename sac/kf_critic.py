from typing import List, Type
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.sac.policies import Critic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class KFCritic(Critic):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
