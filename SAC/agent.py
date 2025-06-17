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

        self.actor = ActorNet(input_shape, actions, hidden_layers, layer_size, activation, continuous=cfg.continuous)
        self.q1 = QNet(input_shape, actions, hidden_layers, layer_size, activation)
        self.q2 = QNet(input_shape, actions, hidden_layers, layer_size, activation)
        self.q1_target = QNet(input_shape, actions, hidden_layers, layer_size, activation)
        self.q2_target = QNet(input_shape, actions, hidden_layers, layer_size, activation)

        self.continuous = cfg.continuous
        self.nets = [self.actor, self.q1, self.q2, self.q1_target, self.q2_target]

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
        }, path + "model_weights.pt")

    def load(self, path: str):
        checkpoint = torch.load(path + "model_weights.pt")
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])

        # Set target networks to eval mode
        self.set_eval()

    def update_target_nets(self, tau=0.01):
        # Update parameters using a moving average
        for new_param, ema_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            ema_param.data.copy_(tau * new_param.data + (1 - tau) * ema_param.data)
        for new_param, ema_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            ema_param.data.copy_(tau * new_param.data + (1 - tau) * ema_param.data)

    def get_action(self, state: torch.Tensor, target: bool = False, scale: bool = True):
        self.set_eval()

        if self.continuous:
            if target:
                act_mean, act_logstd = self.target_actor(state)
            else:
                act_mean, act_logstd = self.actor(state)
            act_std = torch.exp(act_logstd) + cfg.eps  # add small value to avoid log(0)
            actions = act_mean + act_std * torch.randn_like(act_mean)
            
            if cfg.clip is not None:
                # actions = torch.clamp(actions, *cfg.clip)
                actions = torch.tanh(actions)
            if scale:
                actions = actions * cfg.max_action
            return actions, act_mean, act_std
        else:
            if target:
                act_probs, _ = self.target_actor(state)
            else:
                act_probs, _ = self.actor(state)

            action = torch.distributions.Multinomial(
                probs=act_probs
            ).sample().squeeze(0)
            action = torch.argmax(act_probs, dim=-1)

            return action, act_probs, None
    
    
    def set_eval(self):
        [net.eval() for net in self.nets]

    def set_train(self):
        [net.train() for net in self.nets]