import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

hyperparameters = {
    'learning_rate': 0.0003,
    'activation': F.relu,
    'fc1_dims': 256,
    'fc2_dims': 256,
}

class KFCriticNetwork(nn.Module):
    def __init__(self, hparams, input_dims, n_actions,
                 name='critic'):
        super(KHUDA_Finder_CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.name = name

        learning_rate = hparams['learning_rate']
        self.activation_fn = hparams['activation']
        fc1_dims = hparams['fc1_dims']
        fc2_dims = hparams['fc2_dims']

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = self.activation_fn(action_value)
        action_value = self.fc2(action_value)
        action_value = self.activation_fn(action_value)

        q = self.q(action_value)

        return q
