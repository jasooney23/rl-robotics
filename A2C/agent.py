import torch, numpy as np
from torch import nn as nn
from nets import MLP, ActorNet, QNet, ValueNet
import config as cfg

class Agent:
    def __init__(self, input_shape, actions: int, hidden_layers=2, layer_size=64, activation=nn.ELU):
        self.input_shape = input_shape
        self.actions = actions
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.activation = activation

        self.actor = ActorNet(input_shape, actions, hidden_layers, layer_size, activation)
        self.critic = ValueNet(input_shape, hidden_layers, layer_size, activation)
        self.stable_critic = ValueNet(input_shape, hidden_layers, layer_size, activation)
        # self.q1 = QNet(input_shape, actions, hidden_layers, layer_size, activation)
        # self.q2 = QNet(input_shape, actions, hidden_layers, layer_size, activation)

        self.nets = [self.actor, self.critic, self.stable_critic]

    def update_stable_critic(self, tau=0.01):
        # Update parameters using a moving average
        for new_param, ema_param in zip(self.critic.parameters(), self.stable_critic.parameters()):
            ema_param.data.copy_(tau * new_param.data + (1 - tau) * ema_param.data)

    def get_action(self, state: torch.Tensor):
        self.set_eval()
        act_mean, act_std = self.actor(state)
        actions = act_mean + act_std * torch.randn_like(act_mean)
        
        if cfg.clip is not None:
            actions = torch.clamp(actions, *cfg.clip)
        return actions, act_mean, act_std
    
    def set_eval(self):
        [net.eval() for net in self.nets]

    def set_train(self):
        [net.train() for net in self.nets]