import torch, config as cfg
from torch import nn as nn

class ScaledSigmoid(nn.Module):
    def __init__(self):
        super(ScaledSigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * cfg.max_std
    
class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, x):
        return torch.tanh(x) * cfg.max_action