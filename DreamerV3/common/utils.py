import torch, torch.nn as nn
from torch import Tensor
from typing import List
import config as cfg

def calc_cnn_output_size(
    input_size: int,
    kernel_size: List[int],
    stride: List[int],
    padding: List[int],
):
    """ squares only """
    output_size = input_size
    for k, s, p in zip(kernel_size, stride, padding):
        output_size = (output_size - k + 2 * p) // s + 1
    return output_size


def bcall(fn, *args, **kwargs):
    """ Calls a function that expects batched input with a single instance.
        Adds a batch dimension of size 1 to the input and removes it from the output.
        
        Args:
            fn: The function to call.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function."""

    # Add batch dimension to all input tensors
    args = [arg.unsqueeze(0) if isinstance(arg, torch.Tensor) else arg for arg in args]
    kwargs = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

    # Call the function
    output = fn(*args, **kwargs)

    # Remove batch dimension from output tensors
    if isinstance(output, torch.Tensor):
        output = output.squeeze(0)

    # this entire function was written with copilot. i love copilot
    return output


def softmax_kldiv(p: Tensor, q: Tensor):
    '''KL divergence between two latent state softmax distributions'''
    return torch.sum(p * (torch.log(p + cfg.eps) - torch.log(q + cfg.eps)), dim=-1)


def numpify(*args: Tensor):
    return [arg.detach().cpu().to(torch.float32).numpy() for arg in args]


def itemify(*args: Tensor):
    if len(args) == 1:
        return args[0].detach().cpu().item()
    return [arg.detach().cpu().item() for arg in args]


class TanhClamp(nn.Module):
    def __init__(self, min, max):
        super(TanhClamp, self).__init__()
        self.max = max
        self.min = min
 
    def forward(self, x):
        r = self.max - self.min
        c = self.max + self.min
        return torch.tanh(x * 2 / r) * r / 2 + c / 2
    
class UniformProbMix(nn.Module):
    """ Note that this applies only to discrete distributions. Continuous distributions
        don't have a well-defined uniform distribution. Lower bound on variance should
        work instead. """
    def __init__(self, size, uniform_mix_ratio=cfg.uniform_mix_ratio):
        super(UniformProbMix, self).__init__()
        self.size = size
        self.uniform_mix_ratio = uniform_mix_ratio

    def forward(self, x):
        return (1 - self.uniform_mix_ratio) * x + self.uniform_mix_ratio / self.size
    
