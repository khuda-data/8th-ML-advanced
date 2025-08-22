from typing import Optional, List, Type
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.sac.policies import Actor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule


class KFActor(Actor):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = True,
        lr_schedule: Optional[Schedule] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            lr_schedule=lr_schedule,
        )
