import torch, numpy as np, config as cfg
from torch import nn as nn
from agent import Agent
from replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, agent: Agent, 
                 gamma=cfg.gamma, 
                 lr=cfg.lr, 
                 act_ent_weight=cfg.act_ent_weight, 
                 buffer_size=cfg.buffer_size,
                 buffer_dtype=cfg.buffer_dtype,
                 batch_size=cfg.batch_size,):
        self.agent = agent
        self.gamma = gamma
        self.act_ent_weight = act_ent_weight
        self.optimizer = torch.optim.Adam(list(agent.actor.parameters()) + list(agent.critic.parameters()), lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, dtype=buffer_dtype)
        self.batch_size = batch_size

    def loss(self, states, actions, rewards, transitions, dones):
        # s_t
        # a_t: action taken at s_t
        # r_t: reward received after taking action a_t at s_t
        # s_{t+1}: next state after taking action a_t at s_t (transition)
        expected_return = (rewards + self.gamma * self.agent.stable_critic(transitions) * (1 - dones)).detach()
        advantage = (expected_return - self.agent.critic(states))
        
        act_mean, act_std = self.agent.actor(states)
        act_std = act_std + cfg.eps  # to avoid division by zero

        act_probs =  torch.exp(-0.5 * ((actions - act_mean) / act_std) ** 2) / (act_std * np.sqrt(2 * np.pi))
        act_probs = act_probs + cfg.eps  # to avoid log(0)

        actor_target = advantage.detach() * torch.log(act_probs + cfg.eps)
        critic_loss = 0.5 * advantage ** 2
        actor_ent = 0.5 * torch.log(2 * np.pi * act_std ** 2) + 0.5

        actor_target = -actor_target.mean()
        critic_loss = critic_loss.mean()
        actor_ent = -actor_ent.mean()

        loss = actor_target + critic_loss + self.act_ent_weight * actor_ent
        losses = dict(
            actor=actor_target.item(),
            critic=critic_loss.item(),
            actor_ent=actor_ent.item()
        )

        return loss, losses
    
    def add_to_buffer(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state=state, 
            action=action, 
            reward=reward, 
            next_state=next_state, 
            done=done
        )

    def train_step(self):
        self.agent.set_train()

        if self.replay_buffer.size < cfg.buffer_min_size * self.replay_buffer.buffer_size:
            return 0, dict(actor=0, critic=0, actor_ent=0)

        self.optimizer.zero_grad()
        states, actions, rewards, transitions, dones = self.replay_buffer.sample(self.batch_size)
        loss, losses = self.loss(states, actions, rewards, transitions, dones)
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 1)
        nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 1)
        self.optimizer.step()
        self.agent.update_stable_critic(cfg.critic_ema_tau)

        return loss.item(), losses
    
    def train(self):
        total_loss = 0
        losses = dict(actor=0, critic=0, actor_ent=0)
        for i in range(cfg.train_steps_per_update):
            total_loss_i, losses_i = self.train_step()
            total_loss += total_loss_i
            for key in losses:
                losses[key] += losses_i[key]

        total_loss /= cfg.train_steps_per_update
        for key in losses:
            losses[key] /= cfg.train_steps_per_update
        return total_loss, losses