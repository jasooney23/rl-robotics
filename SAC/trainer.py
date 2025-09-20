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
        self.q_optim = torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=lr, weight_decay=cfg.l2_reg_q)
        self.actor_optim = torch.optim.Adam(agent.actor.parameters(), lr=lr, weight_decay=cfg.l2_reg_actor)

        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, dtype=buffer_dtype)
        self.batch_size = batch_size

    def save(self, path: str, train_info=None):
        self.agent.save(path)
        torch.save(self.replay_buffer, path + "replay_buffer.pt")
        torch.save(train_info, path + "train_info.pt")

    def load(self, path: str):
        self.agent.load(path)
        self.replay_buffer = torch.load(path + "replay_buffer.pt", weights_only=False)
        self.replay_buffer.no_grad()
        return torch.load(path + "train_info.pt", weights_only=False)

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
    
    def loss_SAC(self, states_replay, actions_replay, rewards_replay, transitions_replay, dones_replay):
        actions_replay = actions_replay / cfg.max_action  # scale action back to -1 to 1 range for training

        if cfg.continuous:
            actions_next, act_mean_next, act_std_next = self.agent.get_action(transitions_replay, clamp=False)
            actions_curr, act_mean_curr, act_std_curr = self.agent.get_action(states_replay, clamp=False)

            act_unclamped_next = actions_next
            actions_next = torch.tanh(actions_next) # don't multiply; from perspective of agent, action space is -1 to 1. scale only when passing to env
            act_unclamped_probs_next = torch.exp(-0.5 * ((act_unclamped_next - act_mean_next) / act_std_next) ** 2) / (act_std_next * np.sqrt(2 * np.pi))
            act_logprobs_next = torch.log(act_unclamped_probs_next + cfg.eps) - torch.sum(torch.log(1 - actions_next ** 2 + cfg.eps), dim=-1, keepdim=True)

            act_unclamped_curr = actions_curr
            actions_curr = torch.tanh(actions_curr)
            act_unclamped_probs_curr = torch.exp(-0.5 * ((act_unclamped_curr - act_mean_curr) / act_std_curr) ** 2) /(act_std_curr * np.sqrt(2 * np.pi))
            act_logprobs_curr = torch.log(act_unclamped_probs_curr + cfg.eps) - torch.sum(torch.log(1 - actions_curr ** 2 + cfg.eps), dim=-1, keepdim=True)

        else:
            '''ignore this lol'''
            acts, probs, _ = self.agent.get_action(comb_states)
            actions_next, act_probs_next = acts[cfg.batch_size:], probs[cfg.batch_size:]
            actions_curr, curr_act_probs = acts[:cfg.batch_size], probs[:cfg.batch_size]

            # actions_next, act_probs_next, _ = self.agent.get_action(transitions_replay)
            actions_next = nn.functional.one_hot(actions_next.squeeze(), num_classes=cfg.actions_replay).float()
            act_probs_next = actions_next * act_probs_next + cfg.eps

            # actions_curr, curr_act_probs, _ = self.agent.get_action(states_replay)
            actions_curr = nn.functional.one_hot(actions_curr.squeeze(), num_classes=cfg.actions_replay).float()
            curr_act_probs = actions_curr * curr_act_probs + cfg.eps

            actions_replay = nn.functional.one_hot(actions_replay.squeeze(), num_classes=cfg.actions_replay).float()

        # august 10: added detach to actor_target = (q.detach() - self.alpha * act_logprobs_curr)
        # august 12: changed to instead work through different optimizers

        q1_replay = self.agent.q1(states_replay, actions_replay)
        q2_replay = self.agent.q2(states_replay, actions_replay)
        q_target = torch.min(self.agent.q1_target(transitions_replay, actions_next), self.agent.q2_target(transitions_replay, actions_next))
        expected_return = (rewards_replay.unsqueeze(-1) + self.gamma * (1-dones_replay) * (q_target - self.alpha * act_logprobs_next)).detach() # can keep detach here as this won't have grads anyway
        critic_loss = (q1_replay - expected_return) ** 2 + (q2_replay - expected_return) ** 2
        critic_loss = critic_loss.mean()

        self.q_optim.zero_grad()
        critic_loss.backward()
        self.q_optim.step()

        q1_curr = self.agent.q1(states_replay, actions_curr)
        q2_curr = self.agent.q2(states_replay, actions_curr)
        q_curr = torch.min(q1_curr, q2_curr)
        actor_target = (q_curr - self.alpha * act_logprobs_curr)
        actor_loss = -actor_target.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        metrics = dict(
            actor=actor_loss.item(),
            critic=critic_loss.item(),
            avg_std=act_std_curr.mean().item(),
            avg_mean=act_mean_curr.mean().item(),
            max_std=act_std_curr.max().item(),
            max_abs_mean=act_mean_curr.abs().max().item(),

            actor_ent=0  # SAC does not have actor entropy loss
        )

        return actor_loss, critic_loss, metrics

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
            return 0, dict( actor=0, critic=0, actor_ent=0, avg_std=0, avg_mean=0, max_std=0, max_abs_mean=0)

        states, actions, rewards, transitions, dones = self.replay_buffer.sample(self.batch_size)
        actor_loss, critic_loss, metrics = self.loss_SAC(states, actions, rewards, transitions, dones)

        self.agent.update_target_nets(cfg.critic_ema_tau)
        return (actor_loss + critic_loss).item(), metrics

    def train(self):
        total_loss = 0
        metrics = {}
        for i in range(cfg.train_steps_per_update):
            total_loss_i, metrics_i = self.train_step()
            if i == 0:
                metrics = {key: value for key, value in metrics_i.items()}
            else:
                for key in metrics:
                    metrics[key] += metrics_i[key]
            total_loss += total_loss_i
        total_loss /= cfg.train_steps_per_update
        for key in metrics:
            metrics[key] /= cfg.train_steps_per_update
        return total_loss, metrics