from stable_baselines3.sac.policies import MultiInputPolicy
from .kf_actor import KFActor
from .kf_critic import KFCritic


class KFSACPolicy(MultiInputPolicy):
    actor_class = KFActor
    critic_class = KFCritic
