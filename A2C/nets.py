import torch, numpy as np, utils
from torch import nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers, layer_size, activation=nn.ELU, act_last=nn.Identity, batchnorm=False):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.activation = activation
        self.act_last = act_last
        self.batchnorm = batchnorm

        self.nn = nn.ModuleList()

        self.nn.append(nn.Linear(input_shape, layer_size))
        if batchnorm:
            self.nn.append(nn.BatchNorm1d(layer_size))
        self.nn.append(activation())

        for _ in range(hidden_layers - 2):
            self.nn.append(nn.Linear(layer_size, layer_size))
            if batchnorm:
                self.nn.append(nn.BatchNorm1d(layer_size))
            self.nn.append(activation())

        self.nn.append(nn.Linear(layer_size, output_shape))
        self.nn.append(act_last())

    def forward(self, x):
        for i, layer in enumerate(self.nn):
            x = layer(x)
        return x
    
class ActorNet(nn.Module):
    def __init__(self, input_shape, actions: int, hidden_layers=2, layer_size=64, activation=nn.ELU):
        super(ActorNet, self).__init__()
        self.mean_net = MLP(input_shape, actions, hidden_layers, layer_size, activation, act_last=nn.Tanh, batchnorm=True)
        self.std_net = MLP(input_shape, actions, hidden_layers, layer_size, activation, act_last=utils.ScaledSigmoid, batchnorm=True)

    def forward(self, x):
        mean = self.mean_net(x)
    
        std = self.std_net(x)
        return mean, std
    
class QNet(nn.Module):
    def __init__(self, input_shape, actions: int, hidden_layers=2, layer_size=64, activation=nn.ELU):
        super(QNet, self).__init__()
        self.q_net = MLP(input_shape + actions, 1, hidden_layers, layer_size, activation, act_last=nn.Identity, batchnorm=True)

    def forward(self, s, a):
        return self.q_net(torch.cat([s, a], dim=-1))
    
class ValueNet(nn.Module):
    def __init__(self, input_shape, hidden_layers=2, layer_size=64, activation=nn.ELU):
        # stabilized: use moving average for parameters, similar in goal to static target networks
        super(ValueNet, self).__init__()
        self.net = MLP(input_shape, 1, hidden_layers, layer_size, activation, act_last=nn.Identity, batchnorm=True)

    def forward(self, x):
        return self.net(x)
    