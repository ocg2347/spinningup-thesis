import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from typing import List, Union


def get_n_parameters(model):
    return sum(p.numel() for p in model.parameters())


class CNP(nn.Module):
    def __init__(
        self, in_shape, hidden_size, num_hidden_layers, min_std=0.1, encoder_bias=True
    ):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(
            nn.Linear(self.d_x + self.d_y, hidden_size, bias=encoder_bias)
        )
        self.encoder.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(nn.Linear(hidden_size, hidden_size, bias=encoder_bias))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden_size, hidden_size, bias=encoder_bias))
        self.encoder = nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(nn.Linear(hidden_size, hidden_size))
            self.query.append(nn.ReLU())
        self.query.append(nn.Linear(hidden_size, 2 * self.d_y))
        self.query = nn.Sequential(*self.query)

        self.min_std = min_std
        print(self)
        print(get_n_parameters(self))

    def nll_loss(
        self, observation, target, target_truth, observation_mask=None, target_mask=None
    ):
        """
        The original negative log-likelihood loss for training CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.

        Returns
        -------
        loss : torch.Tensor (float)
            TODO: write good doc here
        """
        mean, std = self.forward(observation, target, observation_mask)
        dist = D.Normal(mean, std)
        # print("mean: ", mean, "std: ", std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = nll_masked / nll_norm
        else:
            loss = nll
        if loss.item() > 10.0:
            print("nll loss of cnp:", loss.item())
        return loss

    def forward(self, observation, target, observation_mask=None):
        """
        Forward pass of CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        """
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., : self.d_y]
        logstd = query_out[..., self.d_y :]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(
                dim=1
            )  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(
                1
            )  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(
            1, num_target_points, 1
        )  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class CNP_w_seperateSTD(nn.Module):
    def __init__(
        self,
        in_shape,
        encoder_hidden_sizes,
        decoder_hidden_sizes,
        encoder_output_size=64,
        min_std=0.0,
        encoder_bias=True,
        activation=nn.Tanh,
    ):
        super(CNP_w_seperateSTD, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        log_std = -0.5 * np.ones(in_shape[1], dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.encoder = []
        encoder_layer_widths = [self.d_x + self.d_y] + encoder_hidden_sizes
        for i in range(len(encoder_layer_widths) - 1):
            self.encoder.append(
                nn.Linear(
                    encoder_layer_widths[i],
                    encoder_layer_widths[i + 1],
                    bias=encoder_bias,
                )
            )
            self.encoder.append(activation())
        self.encoder.append(
            nn.Linear(encoder_layer_widths[-1], encoder_output_size, bias=encoder_bias)
        )
        self.encoder = nn.Sequential(*self.encoder)

        self.query = []
        decoder_layer_widths = [encoder_output_size + self.d_x] + decoder_hidden_sizes
        for i in range(len(decoder_layer_widths) - 1):
            self.query.append(
                nn.Linear(decoder_layer_widths[i], decoder_layer_widths[i + 1])
            )
            self.query.append(activation())
        self.query.append(nn.Linear(decoder_layer_widths[-1], self.d_y))
        self.query = nn.Sequential(*self.query)

        self.min_std = min_std
        print(self)
        print("Number of parameters:", get_n_parameters(self))

    def nll_loss(
        self, observation, target, target_truth, observation_mask=None, target_mask=None
    ):
        """
        The original negative log-likelihood loss for training CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.

        Returns
        -------
        loss : torch.Tensor
            (n_batch, n_target, d_y) sized tensor carrying nll loss for each target value
        """
        # print(observation.shape, target.shape)
        # print(observation.shape, target.shape, observation_mask.shape)
        mean, std = self.forward(observation, target, observation_mask)
        dist = D.Normal(mean, std)
        # print("mean: ", mean, "std: ", std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = nll_masked / nll_norm
        else:
            loss = nll
        print(loss.shape)
        # TODO: test and debug this!
        return loss

    def forward(self, observation, target, observation_mask=None):
        """
        Forward pass of CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        """
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        # print(h_cat)
        query_out = self.decode(h_cat)
        mean = query_out[..., : self.d_y]
        std = torch.exp(self.log_std)
        # std = self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(
                dim=1
            )  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(
                1
            )  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(
            1, num_target_points, 1
        )  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class ValueNet(nn.Module):
    def __init__(
        self, obs_dim: int, hidden_sizes: List[int], activation=Union[nn.Tanh, nn.ReLU]
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = []
        self.net.append(nn.Linear(obs_dim, hidden_sizes[0]))
        self.net.append(activation())
        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(activation())
        self.net.append(nn.Linear(hidden_sizes[-1], 1))
        self.net = nn.Sequential(*self.net)
        print(self)
        print(get_n_parameters(self))

    def forward(self, x):
        return torch.squeeze(self.net(x), -1)  # Critical to ensure v has right shape.
