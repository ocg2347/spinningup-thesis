import numpy as np
import scipy.signal
from gym.spaces import Box
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from .models import CNP_w_seperateSTD


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


# Policy Module that uses a Conditional Neural Process to predict the parameters of a Gaussian distribution
class CNPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, encoder_output_size, activation):
        super().__init__()
        self.cnp = CNP_w_seperateSTD(
            in_shape=(obs_dim, act_dim),
            encoder_hidden_sizes=hidden_sizes,
            decoder_hidden_sizes=hidden_sizes,
            activation=activation,
            encoder_output_size=encoder_output_size,
            encoder_bias=True,
        )

    def forward(self, observation, target, target_truth=None, observation_mask=None):
        mean, std = self.cnp(observation, target, observation_mask)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(target_truth)  # (M, 1, d_y)
        return dist, log_prob.sum(axis=-1, keepdim=True)  # [M, 1, 1]

    def _distribution(self, context, obs, observation_mask=None):
        mean, std = self.cnp(
            observation=context, target=obs, observation_mask=observation_mask
        )
        return Normal(mean, std)  # [batch_size, 1, action_dim]

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class CNPActorMLPCritic(nn.Module):
    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(64, 64),
        encoder_output_size=1,
        activation=nn.Tanh,
        use_time_input=True,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        policy_obs_dim = obs_dim + 1 if use_time_input else obs_dim
        if isinstance(action_space, Box):
            self.pi = CNPActor(
                policy_obs_dim,
                action_space.shape[0],
                hidden_sizes,
                encoder_output_size,
                activation,
            )
        else:
            raise NotImplementedError
        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
