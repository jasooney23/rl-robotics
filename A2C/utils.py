import torch, config as cfg
from torch import nn as nn

class ScaledSigmoid(nn.Module):
    def __init__(self):
        super(ScaledSigmoid, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x) * cfg.max_std