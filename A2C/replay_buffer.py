import torch, numpy as np
from torch import nn as nn
import config as cfg

class ReplayBuffer:
    def __init__(self, buffer_size: int = cfg.buffer_size, dtype=torch.float32):
        self.buffer_size = buffer_size
        self.dtype = dtype
        
        self.reset()

    def reset(self):
        self.states = torch.zeros((self.buffer_size, cfg.input_shape), dtype=self.dtype, requires_grad=False)
        self.actions = torch.zeros((self.buffer_size, cfg.actions), dtype=self.dtype, requires_grad=False)
        self.rewards = torch.zeros((self.buffer_size, 1), dtype=self.dtype, requires_grad=False)
        self.next_states = torch.zeros((self.buffer_size, cfg.input_shape), dtype=self.dtype, requires_grad=False)
        self.dones = torch.zeros((self.buffer_size, 1), dtype=self.dtype, requires_grad=False)
        
        self.index = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        
        self.index = (self.index + 1) % self.buffer_size
        if self.size < self.buffer_size:
            self.size += 1

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = self.states[indices].detach()
        actions = self.actions[indices].detach()
        rewards = self.rewards[indices].detach()
        next_states = self.next_states[indices].detach()
        dones = self.dones[indices].detach()

        return states, actions, rewards, next_states, dones