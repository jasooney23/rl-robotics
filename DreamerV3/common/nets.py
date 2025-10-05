import torch, numpy as np, common.utils as utils
from torch import nn as nn
import config as cfg

class MLP(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_layers, 
                 activation=cfg.global_act_fn, act_last=nn.Identity):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.act_last = act_last

        self.nn = nn.ModuleList()

        if hidden_layers is None or len(hidden_layers) == 0:
            self.nn.append(nn.Linear(input_shape, output_shape))
            self.nn.append(act_last())
        else:
            self.nn.append(nn.Linear(input_shape, hidden_layers[0]))
            self.nn.append(activation())
            for i in range(1, len(hidden_layers)):
                self.nn.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
                self.nn.append(activation())
            self.nn.append(nn.Linear(hidden_layers[-1], output_shape))
            self.nn.append(act_last())

    def forward(self, *x):
        x = torch.cat(x, dim=-1) if len(x) > 1 else x[0]
        for i, layer in enumerate(self.nn):
            x = layer(x)
        return x
    
class GaussianMLP(nn.Module):
    def __init__(self, input_shape, output_shape, mlp_layers, 
                 activation=cfg.global_act_fn, mean_bounds=None, logvar_bounds=None):
        
        super(GaussianMLP, self).__init__()
        self.mean_bounds = mean_bounds
        self.logvar_bounds = logvar_bounds
        self.mlp = MLP(input_shape, output_shape*2, mlp_layers, activation)

    def forward(self, *x):
        x = self.mlp(*x)
        mean, logvar = torch.chunk(x, 2, dim=-1)

        if self.mean_bounds is not None:
            mean = utils.TanhClamp(*self.mean_bounds)(mean)
        if self.logvar_bounds is not None:
            logvar = utils.TanhClamp(*self.logvar_bounds)(logvar)

        return mean, logvar
    
    @staticmethod
    def sample(mean, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        return mean + noise * std
