import math
from typing import List

import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)




# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class MultiCategoricalDistribution(nn.Module):
    """
    MultiCategorical distribution for multi discrete actions.
    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, num_inputs, num_outputs, action_dims: List[int]):
        super(MultiCategoricalDistribution, self).__init__()
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

        self.action_dims = action_dims
        self.distributions = None

    def forward(self, x: th.Tensor):
        action_logits = self.linear(x)
        self.distributions = [torch.distributions.Categorical(logits=split) for split in th.split(action_logits, tuple(self.action_dims), dim=1)]
        return self

    def log_probs(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1).unsqueeze(-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1).view(-1, 1)

    def sample(self):
        return torch.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)


class RobotARCategoricalDistribution(nn.Module):
    """
    MultiCategorical distribution for multi discrete actions.
    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, num_inputs, num_outputs, action_dims: List[int]):
        super(RobotARCategoricalDistribution, self).__init__()
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        ae_model = []

        self.action_embedding_dim = 10
        self.a1_logit = nn.Linear(num_inputs, action_dims[0])
        self.joints_embeddings = nn.ModuleList([nn.Sequential(nn.Embedding(d, self.action_embedding_dim), nn.Tanh()) for d in action_dims[:7]]) # only joints
        self.dense_layers = nn.ModuleList([nn.Linear(10 * i, d) for i, d in enumerate(action_dims[1:7], start=1)])

        self.eef = nn.Linear(70, num_outputs - int(np.sum(action_dims[:7])))

        self.action_dims = action_dims
        self.distributions = None

    def forward(self, x: th.Tensor):

        # hardcode for now until can figure algorithm
        a1_logits = self.a1_logit(x)
        dist = torch.distributions.Categorical(logits=a1_logits)
        self.distributions = [dist]
        last_embed = self.joints_embeddings[0](torch.argmax(a1_logits, -1))

        for i in range(6):
            logits = self.dense_layers[i](last_embed)
            dist = torch.distributions.Categorical(logits=logits)
            self.distributions.append(dist)
            action = torch.argmax(logits, -1)
            last_embed = torch.cat([self.joints_embeddings[i+1](action), last_embed], -1)

        end_effector_logits = self.eef(last_embed)
        self.distributions = self.distributions + [torch.distributions.Categorical(logits=split) for split in
                              th.split(end_effector_logits, tuple(self.action_dims[7:]), dim=1)]

        return self

    def log_probs(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1).unsqueeze(-1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1).view(-1, 1)

    def sample(self):
        return torch.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
