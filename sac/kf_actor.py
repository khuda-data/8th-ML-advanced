import torch
import torch.nn as nn
from torch.distributions.normal import Normal

state_size = 36
action_size = 2


class ActorNetwork(nn.Module):
    def __init__(self, layer_size=128):
        super(ActorNetwork, self).__init__()

        self.policy_layers = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
        )

        self.actor_mean = nn.Linear(layer_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_size))

    def forward(self, state):
        features = self.policy_layers(state)

        action_mean = self.actor_mean(features)

        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        policy_dist = Normal(action_mean, action_std)

        return policy_dist
