# sac/kf_policy.py
from typing import Optional, List, Type
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.sac.policies import SACPolicy, Actor, Critic

from kf_actor import KFActor
from kf_critic import KFCritic

class KFSACPolicy(SACPolicy):
 
    actor_class = KFActor
    critic_class = KFCritic

