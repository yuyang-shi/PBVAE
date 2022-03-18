import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from pbvae.models.distributions import independent_normal


class MLP(nn.Module):
    def __init__(self, net_dims, nonlinearity, dropout_p=0.):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.mlp_modules = nn.ModuleList()
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            linear_module = nn.Linear(in_dim, out_dim)
            self.linear_modules.append(linear_module)
            self.mlp_modules.append(linear_module)
            self.mlp_modules.append(nonlinearity)
            self.mlp_modules.append(nn.Dropout(p=dropout_p))
        linear_module = nn.Linear(net_dims[-2], net_dims[-1])
        self.linear_modules.append(linear_module)
        self.mlp_modules.append(linear_module)
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.mlp_modules:
            x = module(x)
        return [x]


class MLP_Normal(nn.Module):
    def __init__(self, net_dims, nonlinearity, dropout_p=0.):
        super(MLP_Normal, self).__init__()
        self.nonlinearity = nonlinearity
        self.mlp_modules = nn.ModuleList()
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            linear_module = nn.Linear(in_dim, out_dim)
            self.linear_modules.append(linear_module)
            self.mlp_modules.append(linear_module)
            self.mlp_modules.append(nonlinearity)
            self.mlp_modules.append(nn.Dropout(p=dropout_p))
        linear_module = nn.Linear(net_dims[-2], net_dims[-1] * 2)
        self.linear_modules.append(linear_module)
        self.mlp_modules.append(linear_module)
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.mlp_modules:
            x = module(x)
        mu, logsigma = torch.chunk(x, 2, dim=-1)
        sig = torch.exp(logsigma)
        return mu, sig


class Constant_Normal(nn.Module):
    def __init__(self, net_dims, init_mu, init_logsigma, fix_mu, fix_logsigma):
        super().__init__()
        if fix_mu:
            self.register_buffer("mu", init_mu * torch.ones(net_dims[-1]))
        else:
            self.mu = nn.Parameter(init_mu * torch.ones(net_dims[-1]))
        if fix_logsigma:
            self.register_buffer("logsigma", init_logsigma * torch.ones(net_dims[-1]))
        else:
            self.logsigma = nn.Parameter(init_logsigma * torch.ones(net_dims[-1]))
        self.net_dims = net_dims

    def forward(self, *args):
        return self.mu, torch.exp(self.logsigma)


class Prob_Linear(nn.Module):
    def __init__(self, in_features, out_features, prior_std, prior_init="normal", sample_network=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sample_network = sample_network

        if prior_init == "zero":
            self.register_buffer("weight_prior_mean", torch.zeros((out_features, in_features)))
            self.register_buffer("bias_prior_mean", torch.zeros(out_features))
        elif prior_init == "normal":
            # Initialise distribution means using clamped normal
            init_std = 1 / np.sqrt(in_features)
            self.register_buffer("weight_prior_mean",
                                 (torch.randn((out_features, in_features)) * init_std).clamp(-2*init_std, 2*init_std))
            self.register_buffer("bias_prior_mean", torch.zeros(out_features))
        else:
            raise NotImplementedError

        self.register_buffer("weight_prior_std", torch.ones(out_features, in_features) * prior_std)
        self.register_buffer("bias_prior_std", torch.ones(out_features) * prior_std)

        self.weight_mean, self.weight_log_std = nn.Parameter(self.weight_prior_mean.clone()), \
                                                nn.Parameter(self.weight_prior_std.clone().log())
        self.bias_mean, self.bias_log_std = nn.Parameter(self.bias_prior_mean.clone()), \
                                            nn.Parameter(self.bias_prior_std.clone().log())
        self.resample_network = True

    def forward(self, input):
        weight_dist = independent_normal(self.weight_mean, self.weight_log_std.exp(), 2)
        bias_dist = independent_normal(self.bias_mean, self.bias_log_std.exp())
        # If self.sample_network is True, during training or during testing and self.resample_network is True
        # Use to train Bayesian networks or test with random sampling
        if self.sample_network and (self.training or self.resample_network):
            weight = weight_dist.rsample()
            bias = bias_dist.rsample()
        # If self.sample_network is True, during testing and self.resample_network is False
        # Use to test Bayesian networks but with fixed sampled network weights
        elif self.sample_network and (not self.training) and (not self.resample_network):
            weight = self.weight
            bias = self.bias
        # If self.sample_network is False
        # Use to train and test deterministic networks
        else:
            weight = self.weight_mean
            bias = self.bias_mean

        return functional.linear(input, weight, bias)

    def compute_prior_kl(self):
        weight_dist = independent_normal(self.weight_mean, self.weight_log_std.exp(), 2)
        bias_dist = independent_normal(self.bias_mean, self.bias_log_std.exp())
        weight_prior = independent_normal(self.weight_prior_mean, self.weight_prior_std, 2)
        bias_prior = independent_normal(self.bias_prior_mean, self.bias_prior_std)
        return torch.distributions.kl.kl_divergence(weight_dist, weight_prior) + \
               torch.distributions.kl.kl_divergence(bias_dist, bias_prior)

    def init_prior_module(self, prior_module, init_weights_with_prior=True):
        self.weight_prior_mean.data = prior_module.weight.data.clone()
        self.bias_prior_mean.data = prior_module.bias.data.clone()
        if init_weights_with_prior:
            self.weight_mean.data = prior_module.weight.data.clone()
            self.bias_mean.data = prior_module.bias.data.clone()

    def resample(self):
        # assert not self.training
        weight_dist = independent_normal(self.weight_mean, self.weight_log_std.exp(), 2)
        bias_dist = independent_normal(self.bias_mean, self.bias_log_std.exp())
        self.weight = weight_dist.sample()
        self.bias = bias_dist.sample()

    def set_sample_network(self, mode):
        self.sample_network = mode

    def set_resample_network(self, mode):
        self.resample_network = mode
    
    def set_weight_std(self, weight_std):
        self.weight_log_std.data = torch.ones_like(self.weight_log_std.data) * np.log(weight_std)
        self.bias_log_std.data = torch.ones_like(self.bias_log_std.data) * np.log(weight_std)


class Prob_MLP(nn.Module):
    def __init__(self, net_dims, nonlinearity, prior_std, prior_init="normal", sample_network=True):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.mlp_modules = nn.ModuleList()
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            linear_module = Prob_Linear(in_dim, out_dim, prior_std, prior_init=prior_init, sample_network=sample_network)
            self.linear_modules.append(linear_module)
            self.mlp_modules.append(linear_module)
            self.mlp_modules.append(nonlinearity)
        linear_module = Prob_Linear(net_dims[-2], net_dims[-1], prior_std, prior_init=prior_init, sample_network=sample_network)
        self.linear_modules.append(linear_module)
        self.mlp_modules.append(linear_module)
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.mlp_modules:
            x = module(x)
        return [x]

    def compute_prior_kl(self):
        kl_sum = 0
        for linear_module in self.linear_modules:
            kl_sum += linear_module.compute_prior_kl()
        return kl_sum

    def init_prior_module(self, prior_module, init_weights_with_prior):
        assert len(self.linear_modules) == len(prior_module.linear_modules)
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].init_prior_module(prior_module.linear_modules[i], init_weights_with_prior=init_weights_with_prior)

    def resample(self):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].resample()

    def set_sample_network(self, mode):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_sample_network(mode)

    def set_resample_network(self, mode):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_resample_network(mode)
    
    def set_weight_std(self, weight_std):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_weight_std(weight_std)


class Prob_MLP_Normal(nn.Module):
    def __init__(self, net_dims, nonlinearity, prior_std, prior_init="normal", sample_network=True):
        super().__init__()
        self.nonlinearity = nonlinearity
        self.mlp_modules = nn.ModuleList()
        self.linear_modules = nn.ModuleList()
        for in_dim, out_dim in zip(net_dims[:-2], net_dims[1:-1]):
            linear_module = Prob_Linear(in_dim, out_dim, prior_std, prior_init=prior_init, sample_network=sample_network)
            self.linear_modules.append(linear_module)
            self.mlp_modules.append(linear_module)
            self.mlp_modules.append(nonlinearity)
        linear_module = Prob_Linear(net_dims[-2], net_dims[-1] * 2, prior_std, prior_init=prior_init, sample_network=sample_network)
        self.linear_modules.append(linear_module)
        self.mlp_modules.append(linear_module)
        self.net_dims = net_dims

    def forward(self, x):
        for module in self.mlp_modules:
            x = module(x)
        mu, logsigma = torch.chunk(x, 2, dim=-1)
        sig = torch.exp(logsigma)
        return mu, sig
    
    def compute_prior_kl(self):
        kl_sum = 0
        for linear_module in self.linear_modules:
            kl_sum += linear_module.compute_prior_kl()
        return kl_sum

    def init_prior_module(self, prior_module, init_weights_with_prior):
        assert len(self.linear_modules) == len(prior_module.linear_modules)
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].init_prior_module(prior_module.linear_modules[i], init_weights_with_prior=init_weights_with_prior)

    def resample(self):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].resample()

    def set_sample_network(self, mode):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_sample_network(mode)

    def set_resample_network(self, mode):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_resample_network(mode)
    
    def set_weight_std(self, weight_std):
        for i in range(len(self.linear_modules)):
            self.linear_modules[i].set_weight_std(weight_std)

