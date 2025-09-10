import torch, config as cfg
from torch import nn as nn

class ScaledSigmoid(nn.Module):
    def __init__(self, max):
        super(ScaledSigmoid, self).__init__()
        self.max = max

    def forward(self, x):
        return torch.sigmoid(x) * self.max
    
class ScaledTanh(nn.Module):
    def __init__(self, max):
        super(ScaledTanh, self).__init__()
        self.max = max

    def forward(self, x):
        return torch.tanh(x) * self.max
    
class TanhClamp(nn.Module):
    def __init__(self, min, max):
        super(TanhClamp, self).__init__()
        self.max = max
        self.min = min
 
    def forward(self, x):
        r = self.max - self.min
        c = self.max + self.min
        return torch.tanh(x * 2 / r) * r / 2 + c / 2