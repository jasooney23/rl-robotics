import torch, numpy as np, config as cfg
from torch import nn as nn
from agent import Agent
from replay_buffer import ReplayBuffer

class Trainer:
    def __init__(self, agent: Agent, 
                 gamma=cfg.gamma, 
                 lr=cfg.lr, 
                 alpha=cfg.alpha, 
                 buffer_size=cfg.buffer_size,
                 buffer_dtype=cfg.buffer_dtype,
                 batch_size=cfg.batch_size,):
        self.agent = agent
        self.gamma = gamma
        self.alpha = alpha
        params = []
        for net in agent.nets:
            params += list(net.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, dtype=buffer_dtype)
        self.batch_size = batch_size

    def save(self, path: str):
        self.agent.save(path)
        torch.save(self.replay_buffer, path + "replay_buffer.pt")

    def load(self, path: str):
        self.agent.load(path)
        self.replay_buffer = torch.load(path + "replay_buffer.pt")
        self.replay_buffer.no_grad()

    def loss_A2C(self, states, actions, rewards, transitions, dones):
        # s_t
        # a_t: action taken at s_t
        # r_t: reward received after taking action a_t at s_t
        # s_{t+1}: next state after taking action a_t at s_t (transition)
        target_act_next, _, _ = self.agent.get_action(transitions, target=True)
        if not cfg.continuous:
            actions = nn.functional.one_hot(actions.squeeze(), num_classes=cfg.actions).float()
            target_act_next = nn.functional.one_hot(target_act_next, num_classes=cfg.actions).float()
        expected_return = (rewards.unsqueeze(-1) + self.gamma * self.agent.target_critic(transitions, target_act_next) * (1 - dones)).detach()
        advantage = (expected_return - self.agent.critic(states, actions))
        
        critic_loss = 0.5 * advantage ** 2
        if cfg.continuous:
            act_mean, act_logstd = self.agent.actor(states)
            act_std = torch.exp(act_logstd)

            act_probs =  torch.exp(-0.5 * ((actions - act_mean) / act_std) ** 2) / (act_std * np.sqrt(2 * np.pi))
            act_probs = act_probs + cfg.eps  # to avoid log(0)
            actor_ent = 0.5 * torch.log(2 * np.pi * act_std ** 2) + 0.5
        else:
            act_probs, _ = self.agent.actor(states)
            actor_ent = -1 * torch.sum(act_probs * torch.log(act_probs + cfg.eps), dim=-1)

        actor_target = (advantage.detach() * torch.log(act_probs + cfg.eps)) * actions

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
    
    def loss_SAC(self, states, actions, rewards, transitions, dones):
        if cfg.continuous:
            actions_next, act_mean_next, act_std_next = self.agent.get_action(transitions, scale=False)
            act_unclamped_next = torch.tanh(actions_next)
            act_unclamped_probs_next = torch.exp(-0.5 * ((act_unclamped_next - act_mean_next) / act_std_next) ** 2) \
                / (act_std_next * np.sqrt(2 * np.pi))  # to avoid log(0)
            act_logprobs_next = torch.log(act_unclamped_probs_next + cfg.eps) - torch.sum(torch.log(1 - actions_next ** 2 + cfg.eps), dim=-1, keepdim=True)

            curr_actions, curr_act_mean, curr_act_std = self.agent.get_action(states, scale=False)
            curr_act_unclamped = torch.tanh(curr_actions)
            curr_act_unclamped_probs = torch.exp(-0.5 * ((curr_act_unclamped - curr_act_mean) / curr_act_std) ** 2) \
                / (curr_act_std * np.sqrt(2 * np.pi))  # to avoid log(0)
            curr_logprobs_next = torch.log(curr_act_unclamped_probs + cfg.eps) - torch.sum(torch.log(1 - curr_actions ** 2 + cfg.eps), dim=-1, keepdim=True)
        else:
            actions_next, act_probs_next, _ = self.agent.get_action(transitions)
            actions_next = nn.functional.one_hot(actions_next.squeeze(), num_classes=cfg.actions).float()
            act_probs_next = actions_next * act_probs_next + cfg.eps

            curr_actions, curr_act_probs, _ = self.agent.get_action(states)
            curr_actions = nn.functional.one_hot(curr_actions.squeeze(), num_classes=cfg.actions).float()
            curr_act_probs = curr_actions * curr_act_probs + cfg.eps

            actions = nn.functional.one_hot(actions.squeeze(), num_classes=cfg.actions).float()

        q1_target = self.agent.q1_target(transitions, actions_next)
        q2_target = self.agent.q2_target(transitions, actions_next)
        q_target = torch.min(q1_target, q2_target)
        expected_return = (rewards.unsqueeze(-1) + self.gamma * (1-dones) * (q_target - self.alpha * act_logprobs_next)).detach()

        critic_loss = (self.agent.q1(states, actions) - expected_return) ** 2 

        q1 = self.agent.q1(states, curr_actions)
        q2 = self.agent.q2(states, curr_actions)
        q = torch.min(q1, q2)

        actor_target = (q - self.alpha * curr_logprobs_next)

        actor_target = -actor_target.mean()
        critic_loss = critic_loss.mean()

        loss = actor_target + critic_loss
        losses = dict(
            actor=actor_target.item(),
            critic=critic_loss.item(),
            actor_ent=0  # SAC does not have actor entropy loss
        )
        if torch.isnan(loss).any():
            raise ValueError("NaN detected in actions")
        return loss, losses, 

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
        loss, losses = self.loss_SAC(states, actions, rewards, transitions, dones)
        loss.backward()
        for net in self.agent.nets:
            nn.utils.clip_grad_norm_(net.parameters(), cfg.gradnorm_clip)
        self.optimizer.step()
        act, act_mean, act_std = self.agent.get_action(states)
        if torch.isnan(act).any():
            raise ValueError("NaN detected in actions")
        self.agent.update_target_nets(cfg.critic_ema_tau)

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