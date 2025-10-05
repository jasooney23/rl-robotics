'''In version 3:
 - Reverting back to deter only for dyn() and repr();
   predicting it will work better this time since stoch
   is now discrete, perhaps the behaviour will be different.
'''

import torch, time, os, torchviz
from torch import nn
import numpy as np, torch.nn.functional as F

import config as cfg, wm.replay_buffer
from common.nets import MLP, GaussianMLP
from common import utils


class StraightThroughSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """ input is softmax'd (N, size), output is (N, 1) and sampled """
        sampled = torch.multinomial(input, num_samples=1).squeeze(-1) # sample to flat
        return sampled
    @staticmethod
    def backward(ctx, grad_output):
        """ straight-through gradient """
        return grad_output # might not work

class SampleSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """ input is softmax'd (N, z_size, z_size), output is (N, z_size ** 2) and sampled """
        N = input.shape[0]
        z_size = int(input.shape[1])
        input = input.view(N * z_size, z_size)

        sampled = torch.multinomial(input, num_samples=1).squeeze(-1) # sample to flat
        one_hot = F.one_hot(sampled, num_classes=input.size(1)).to(cfg.data_dtype) # one-hot to unflat into square
        one_hot = one_hot.view(-1, input.size(1) ** 2) # flatten
        return one_hot
    @staticmethod
    def backward(ctx, grad_output):
        """ straight-through gradient """
        z_size = int(grad_output.size(1) ** 0.5)
        return grad_output.view(-1, z_size, z_size)


class SequentialModel(nn.Module):
    '''ht = deter(ht-1, st-1, at-1). Deterministic.
       This might have to go up in complexity, the GRU used here is different from original model.'''

    def __init__(self, h_size, z_size, act_size, gru_layers=cfg.GRU_layers):
        super(SequentialModel, self).__init__()

        input_size = h_size + z_size ** 2 + act_size
        self.gru = nn.GRU(input_size, cfg.h_size, num_layers=gru_layers, batch_first=False)
        self.n_layers = gru_layers

        if gru_layers > 1:
            raise NotImplementedError("GRU layers > 1 not implemented yet")
    
    def forward(self, h, z, a):
        ''' 1 layer GRU only, so the hidden state is same as output h
        Args:
           h: (N, h_size) -> (N, 1, h_size)
           z: (N, z_size ** 2) -> (N, 1, z_size ** 2)
           a: (N, act_size) -> (N, 1, act_size)'''
        # N = batch size
        # L = 1, seq length
        # H_in = h_size + z_size ** 2 + act_size, input size
        # H_out = h_size, hidden size
        inp = torch.cat((h, z, a), dim=-1).unsqueeze(0) # (1, N, H_in)
        h = h.unsqueeze(0) # (1, N, H_out)
        _, h = self.gru(inp, h)
        h = h.squeeze(0) # (N, H_out)

        return h 


# CNN, returns prob dist
class LatentEncoder(nn.Module):
    """ ENFORCE THAT THE MLP BEFORE/AFTER CONV/DECONV STAGES MAINTAINS DIMENSIONALITY"""
    def __init__(self, h_size, z_size, obs_size, mlp_layers=cfg.mlp_layers, batchnorm=False):
        """ obs_size is (C, H, W) """

        super(LatentEncoder, self).__init__()

        self.batchnorm = batchnorm
        self.act_fn = cfg.global_act_fn()
        self.unimix = utils.UniformProbMix(z_size, cfg.uniform_mix_ratio)

        # calculate convolution output sizes
        self.conv = nn.ModuleList([nn.Conv2d(cfg.channels[0][i-1], 
                                             cfg.channels[0][i], 
                                             cfg.kernels[0][i], 
                                             stride=cfg.strides[0][i], 
                                             padding=cfg.paddings[0][i]) for i in range(1, len(cfg.channels[0]))])
        self.conv.insert(0, nn.Conv2d(obs_size[0], cfg.channels[0][0], cfg.kernels[0][0], stride=cfg.strides[0][0], padding=cfg.paddings[0][0]))
        self.norm = nn.ModuleList([nn.BatchNorm2d(cfg.channels[0][i]) if batchnorm else nn.Identity() for i in range(len(cfg.channels[0]))])
        
        self.flatten = nn.Flatten()

        cnn_out_size = utils.calc_cnn_output_size(obs_size[1], cfg.kernels[0], cfg.strides[0], cfg.paddings[0])
        cnn_out_size = (cnn_out_size ** 2) * cfg.channels[0][-1] # final conv layer output size
        self.mlp = MLP(cnn_out_size + h_size, z_size ** 2, mlp_layers)
        self.h_size = h_size
        self.z_size = z_size
        self.cnn_out_size = cnn_out_size


    def forward(self, h, x, return_probs=False):
        for i in range(len(self.conv)):
            x = self.act_fn(self.conv[i](x))
            x = self.norm[i](x)

        x = self.flatten(x)

        x = torch.cat((x, h), 1)
        x = self.mlp(x)

        x = x.view(-1, self.z_size, self.z_size) # square
        x = torch.softmax(x, dim=2) # softmax-over-classes (gumbel softmax)
        probs = self.unimix(x) # mix with uniform distribution
        x = SampleSoftmax.apply(probs) # straight-through sample

        if return_probs:
            return x, probs
        return x
    

class LatentDecoder(nn.Module):
    def __init__(self, h_size, z_size, mlp_layers=cfg.mlp_layers, batchnorm=False):
        super(LatentDecoder, self).__init__()
        self.conv_size = np.array((z_size ** 2, 1, 1))

        self.mlp = MLP(h_size + z_size ** 2, np.prod(self.conv_size), mlp_layers)
        self.convT = nn.ModuleList([nn.ConvTranspose2d(cfg.channels[1][i-1], 
                                                       cfg.channels[1][i], 
                                                       cfg.kernels[1][i], 
                                                       stride=cfg.strides[1][i], 
                                                       padding=cfg.paddings[1][i], 
                                                       output_padding=0) for i in range(1, len(cfg.channels[1]))])
        self.convT.insert(0, nn.ConvTranspose2d(np.prod(self.conv_size), cfg.channels[1][0], cfg.kernels[1][0], stride=cfg.strides[1][0], padding=cfg.paddings[1][0], output_padding=0))
        self.norm = nn.ModuleList([nn.BatchNorm2d(cfg.channels[1][i]) if batchnorm else nn.Identity() for i in range(len(cfg.channels[1]))])

        self.act_fn = cfg.global_act_fn()

    def forward(self, h, z):
        x = torch.cat((h, z), 1)
        x = self.mlp(x)
        x = x.view(-1, *self.conv_size)

        for i in range(len(self.convT) - 1):
            x = self.act_fn(self.convT[i](x))
            x = self.norm[i](x)
        x = torch.sigmoid(self.convT[-1](x)) # constrain to [0, 1] for images. Later scaled to [0, 255]

        return x


class DynamicPredictor(nn.Module):
    def __init__(self, h_size, z_size, mlp_layers=cfg.mlp_layers):
        super(DynamicPredictor, self).__init__()

        self.mlp = MLP(h_size, z_size ** 2, mlp_layers)
        self.z_size = z_size
        self.unimix = utils.UniformProbMix(z_size, cfg.uniform_mix_ratio)

    def forward(self, h, return_probs=False):
        '''Returns as probabilities'''

        # output is mean, log variance for a Gaussian
        x = self.mlp(h)
        x = x.view(-1, self.z_size, self.z_size)
        x = torch.softmax(x, dim=2)
        probs = self.unimix(x)
        x = SampleSoftmax.apply(probs) # straight-through sample

        if return_probs:
            return x, probs
        return x
    

class RewardPredictor(nn.Module):
    def __init__(self, h_size, z_size, mlp_layers=cfg.mlp_layers):
        super(RewardPredictor, self).__init__()

        self.gaussian = GaussianMLP(h_size + z_size ** 2, 1, mlp_layers, mean_bounds=cfg.rew_mean_bounds, logvar_bounds=cfg.rew_logvar_bounds)

    def forward(self, h, z, return_probs=False):
        x = torch.cat((h, z), 1)
        mean, logvar = self.gaussian(x)
        samples = GaussianMLP.sample(mean, logvar)

        if return_probs:
            return samples, (mean, logvar)
        return samples


class TerminationPredictor(nn.Module):
    def __init__(self, h_size, z_size, mlp_layers=cfg.mlp_layers):
        super(TerminationPredictor, self).__init__()
        self.mlp = MLP(h_size + z_size ** 2, 2, mlp_layers, act_last=nn.Softmax)

    def forward(self, h, z):
        x = torch.cat((h, z), 1)
        x = self.mlp(x)
        x = StraightThroughSample.apply(x)
        x = x.view(-1, 1)
        # test
        assert len(x.shape) == 2
        assert x.shape[1] == 1
        return x