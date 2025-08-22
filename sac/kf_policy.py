from stable_baselines3.sac.policies import SACPolicy
from kf_actor import KFActor
from kf_critic import KFCritic


class KFSACPolicy(SACPolicy):
    actor_class = KFActor
    critic_class = KFCritic
