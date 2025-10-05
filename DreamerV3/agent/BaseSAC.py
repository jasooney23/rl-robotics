import torch, numpy as np
from torch import nn as nn
from torch import Tensor
from typing import List

import config as cfg, common.utils as utils
from common.nets import MLP, GaussianMLP
from common.utils import bcall


from agent.Agent import Agent, AgentTrainer
from wm.RSSM import RSSM
class BaseSAC(Agent):
    """ Base SAC algorithm as specified in original paper. """
    def __init__(self, input_shape, actions: int, mlp_layers: List[int]=cfg.mlp_layers, activation=cfg.global_act_fn):
        super().__init__(input_shape=input_shape, 
                         actions=actions)
        self.mlp_layers = mlp_layers
        self.activation = activation

        self.actor = GaussianMLP(input_shape, actions, mlp_layers, activation).to(cfg.device).to(cfg.data_dtype)
        self.q1 = MLP(input_shape + actions, 1, mlp_layers, activation).to(cfg.device).to(cfg.data_dtype)
        self.q2 = MLP(input_shape + actions, 1, mlp_layers, activation).to(cfg.device).to(cfg.data_dtype)
        self.q1_target = MLP(input_shape + actions, 1, mlp_layers, activation).to(cfg.device).to(cfg.data_dtype)
        self.q2_target = MLP(input_shape + actions, 1, mlp_layers, activation).to(cfg.device).to(cfg.data_dtype)

        self.nets = [self.actor, self.q1, self.q2, self.q1_target, self.q2_target]
        self.q_nets = [self.q1, self.q2]
        self.q_target_nets = [self.q1_target, self.q2_target]

    def q(self, state: Tensor, action: Tensor) -> Tensor:
        """ Return Q-value with minimization trick to reduce overestimation bias. """
        return torch.min(self.q1(state, action), self.q2(state, action))


    def q_target(self, state: Tensor, action: Tensor) -> Tensor:
        """ Return target Q-value with minimization trick to reduce overestimation bias. """
        return torch.min(self.q1_target(state, action), self.q2_target(state, action))


    def update_target_nets(self, tau=0.01):
        """ Update parameters using a moving average """
        with torch.no_grad():
            for q, q_target in zip(self.q_nets, self.q_target_nets):
                for new_param, ema_param in zip(q.parameters(), q_target.parameters()):
                    ema_param.mul_(1 - tau)
                    ema_param.add_(tau * new_param.data)


    def get_action(self, state: Tensor, train_mode: bool=False, scale: bool=True) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """ Sample action from policy, clamp = true when sampling for environment step, false
            when sampling for training step. Batched inputs & outputs only.
        Args:
            state: Latent state. Is concat of (h, z) if using RSSM. Shape (input_shape,)
            train_mode: Enables set_train()
            scale: Apply action scale or keep at [-1, 1]
        Returns:
            action: Action sampled from policy. Shape (N, actions,)
            act_mean: Mean of action distribution. Shape (N, actions,)
            act_std: Standard deviation of action distribution. Shape (N, actions,)
            act_logprobs: Log probability density of sampled actions. Shape (N, actions,)
        """
        self.set_train() if train_mode else self.set_eval()

        act_mean, act_logvar = self.actor(state)
        act_logvar = utils.TanhClamp(*cfg.act_logvar_bounds)(act_logvar)  # clamp log var to avoid extreme values
        act_std = torch.exp(0.5 * act_logvar) + cfg.eps  # add small value to avoid log(0)
        if cfg.raw_mean_range is not None:
            act_mean = utils.TanhClamp(cfg.raw_mean_range[0], cfg.raw_mean_range[1])(act_mean)  # clamp raw mean to avoid extreme values
        noise = torch.randn_like(act_mean)
        act_raw = act_mean + act_std * noise  # reparam trick

        act_raw_probs = torch.exp(-0.5 * ((act_raw - act_mean) / act_std) ** 2) /(act_std * np.sqrt(2 * np.pi))
        actions = torch.tanh(act_raw) 
        act_logprobs = torch.log(act_raw_probs + cfg.eps) - torch.sum(torch.log(1 - actions ** 2 + cfg.eps), dim=-1, keepdim=True)

        if act_logprobs.isnan().any():
            raise ValueError("screw you")
               
        actions = actions * cfg.max_action if scale else actions

        aux = dict(
            act_raw=act_raw,
            act_logprob=act_logprobs,
        )

        return actions, act_mean, act_std, aux


    def get_step_action(self, state: Tensor) -> Tensor:
        """ Shortcut to get actions for an environment step. Single input only.
        Args:
            state: latent state. Is concat of (h, z) if using RSSM. Shape  (input_shape,)
        Returns:
            actions: Action sampled from policy. Shape (actions,)
        """
        self.set_eval()
        if len(state.shape) == 1:
            actions = bcall(self.get_action, state)[0]
        else:
            raise ValueError("State must be shape (input_shape,)")
        return actions


    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
        }, path + "model_weights.pt")
        print(f"BaseSAC: Saved model weights to {path}model_weights.pt")


    def load(self, path: str):
        checkpoint = torch.load(path + "model_weights.pt", weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])

        # Set target networks to eval mode
        self.set_eval()
        print(f"BaseSAC: Loaded model weights from {path}model_weights.pt")


class BaseSAC_Trainer(AgentTrainer):
    def __init__(self, agent: Agent, 
                 world_model: RSSM,
                 gamma=cfg.gamma, 
                 lr=cfg.lr, 
                 alpha=cfg.alpha, 
                 clip_grad=cfg.grad_norm_clip,
                 batch_size=cfg.batch_size,
                 subseq_len=cfg.subseq_len):
        super().__init__(agent)
        self.agent = agent
        self.world_model = world_model
        self.gamma = gamma
        self.alpha = alpha
        self.clip_grad = clip_grad
        self.batch_size = batch_size
        self.subseq_len = subseq_len

        # separate optimizers for separate gradient updates
        self.actor_params = list(self.agent.actor.parameters())
        self.q_params = list(self.agent.q1.parameters()) + list(self.agent.q2.parameters())
        self.q_optim = torch.optim.Adam(self.q_params, lr=lr, weight_decay=cfg.l2_reg)
        self.actor_optim = torch.optim.Adam(self.actor_params, lr=lr, weight_decay=cfg.l2_reg)


    def train_on_subseq(self):
        """ Train on a batch of trajectories jointly generated by the replay buffer & the agent.

        Steps:
        1. Generate the trajectories as leaves (no gradients attached)
        2. Calculate target return estimates for each step in the trajectory (backwards in time)
        3. Flatten trajectories to just be a batch of (s, a, r, s', t) tuples
        4. Train actor & critic as a normal batch gradient step
        """

        batch_init = self.world_model.buffer.sample(self.batch_size, init_step_only=True)
        batch_traj = self.generate_trajectory(batch_init)
        batch_traj["ret"] = self.calc_traj_returns(batch_traj)

        self.agent.set_train() # generate trajectory uses eval for agent & WM

        for k in batch_traj.keys():
            batch_traj[k] = batch_traj[k].view(-1, *batch_traj[k].shape[2:]) # flatten (N, T, *) to (NT, *)

        loss_total, losses, metrics = self.gradient_step(batch_traj)
        self.agent.update_target_nets()

        return loss_total, losses, metrics


    def generate_trajectory(self, batch_init: dict[str, Tensor]) -> dict[str, Tensor]:
        """ Generate a batch of trajectories using the current policy and world model. No gradients.
            Args:
                batch_init: A dictionary of initial states, one for each signal. Each tensor is shape (N, 1, *) for batch size N.
            Returns:
                A dictionary of generated trajectories with signals (s, a, r, d). Each tensor is shape (N, T, *) for batch size N and trajectory length T.
        """
        self.agent.set_eval()
        self.world_model.set_eval()

        agent = self.agent
        rssm = self.world_model

        N = self.batch_size
        T = self.subseq_len

        with torch.no_grad():
            traj = dict(
                state=torch.empty((N, T, self.agent.input_shape), device=cfg.device, dtype=cfg.data_dtype), 
                act=torch.empty((N, T, self.agent.actions), device=cfg.device, dtype=cfg.data_dtype), 
                act_raw=torch.empty((N, T, self.agent.actions), device=cfg.device, dtype=cfg.data_dtype),
                act_logprob=torch.empty((N, T, 1), device=cfg.device, dtype=cfg.data_dtype),
                rew=torch.empty((N, T, 1), device=cfg.device, dtype=cfg.data_dtype), 
                done=torch.empty((N, T, 1), device=cfg.device, dtype=cfg.data_dtype),
            )
            for t in range(T):
                if t == 0:
                    ht = batch_init["h"].squeeze(1)
                else:
                    ht = rssm.seq(ht, zt, at)
                zt = rssm.dyn(ht).view(-1, rssm.z_size ** 2)  # (N, z_size ** 2)
                st = torch.cat((ht, zt), dim=-1)
                at, _, _, aux = agent.get_action(st, scale=False)
                rt = rssm.rew(ht, zt)
                dt = rssm.don(ht, zt)

                traj["state"][:, t] = st
                traj["act"][:, t] = at
                traj["act_raw"][:, t] = aux["act_raw"]
                traj["act_logprob"][:, t] = aux["act_logprob"]
                traj["rew"][:, t] = rt
                traj["done"][:, t] = dt

        return traj


    def calc_traj_returns(self, traj: dict[str, Tensor]) -> Tensor:
        """ 
            Calculate the return estimates for each step in the trajectory. No gradients.
            Args:
                traj: A dictionary of generated trajectories with signals (s, a, r, d). Each tensor is shape (N, T, *) for batch size N and trajectory length T.
            Returns:
                A tensor of return estimates for each step in the trajectory. Shape (N, T, 1)
        """
        self.agent.set_eval()
        N = traj["state"].shape[0]
        T = traj["state"].shape[1]

        with torch.no_grad():
            returns = torch.zeros((N, T, 1), dtype=cfg.data_dtype, device=traj["state"].device)
            for t in reversed(range(T)):
                if t == T - 1:
                    # i THINK it is correct to use the sampled action from the trajectory to estimate v(s).
                    #     v(s) = E_(a~pi)[q - ln(pi)]. and i also THINK it is correct to only use 1 sample
                    #     to estimate the expectation, since SAC does this too.
                    # There might be some benefit I suppose to having multiple samples as monte-carlo estimation,
                    #     but idk what benefits that would have.
                    Rt = self.agent.q_target(traj["state"][:, t], traj["act"][:, t]) + self.alpha * traj["act_logprob"][:, t]
                else:
                    Rt = traj["rew"][:, t] + self.gamma * (1 - traj["done"][:, t]) * Rt
                returns[:, t] = Rt

        if returns.isnan().any():
            raise ValueError("screw you")

        return returns


    def gradient_step(self, batch_traj: dict) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """ Compute SAC losses for a batch of transitions AND apply gradients"""
        states_replay = batch_traj["state"]
        actions_replay = batch_traj["act"]
        act_raw_replay = batch_traj["act_raw"]
        returns_replay = batch_traj["ret"]
        

        # critic loss
        q1_replay = self.agent.q1(states_replay, actions_replay)
        q2_replay = self.agent.q2(states_replay, actions_replay)
        critic_loss = (q1_replay - returns_replay) ** 2 + (q2_replay - returns_replay) ** 2
        critic_loss = critic_loss.mean()
        
        critic_loss = critic_loss * cfg.loss_stability_scale
        self.q_optim.zero_grad() # this step has to happen before actor update,
                                 #     or else the parameters will have changed 
                                 #     in the graph and torch doesn't like that
        critic_loss.backward()
        for p in self.q_params: # scale up loss since bfloat16 has lower decimal precision, then scale back down gradients to be consistent
            if p.grad is not None:
                p.grad.data = p.grad.data / cfg.loss_stability_scale
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.q_params, self.clip_grad)
        self.q_optim.step()

        critic_loss = critic_loss.detach() / cfg.loss_stability_scale


        # actor loss
        _, actions_mean, actions_std, _ = self.agent.get_action(states_replay, train_mode=True, scale=False)
        action_probs = torch.exp(-0.5 * ((act_raw_replay - actions_mean) / actions_std) ** 2) / (actions_std * np.sqrt(2 * np.pi))
        action_logprobs = torch.log(action_probs + cfg.eps) - torch.sum(torch.log(1 - actions_replay ** 2 + cfg.eps), dim=-1, keepdim=True)

        n_act = actions_replay.shape[-1]
        action_entropy = (n_act / 2) * np.log(2 * np.pi * np.e) + torch.sum(torch.log(actions_std + cfg.eps), dim=-1, keepdim=True)
        actor_target = returns_replay * action_logprobs + self.alpha * action_entropy
        actor_loss = -actor_target.mean()

        actor_loss = actor_loss * cfg.loss_stability_scale
        self.actor_optim.zero_grad() # STEP
        actor_loss.backward()
        for p in self.actor_params: # scale up loss since bfloat16 has lower decimal precision, then scale back down gradients to be consistent
            if p.grad is not None:
                p.grad.data = p.grad.data / cfg.loss_stability_scale
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.actor_params, self.clip_grad)
        self.actor_optim.step()

        actor_loss = actor_loss.detach() / cfg.loss_stability_scale

        TESTACT = self.agent.get_action(states_replay, train_mode=True, scale=False)[0]
        if torch.isnan(TESTACT).any():
            raise ValueError("screw you")


        losses = dict(
            AGENT_actor=actor_loss,
            AGENT_critic=critic_loss
        )
        metrics = dict(
            # these metrics are cool
            AGENT_actor_avg_std=actions_std.mean(),
            AGENT_actor_avg_mean=actions_mean.mean(),
            AGENT_actor_max_std=actions_std.max(),
            AGENT_actor_max_abs_mean=actions_mean.abs().max(),
        )

        loss_total = actor_loss + critic_loss

        return loss_total, losses, metrics


    def save(self, path: str):
        self.agent.save(path)


    def load(self, path: str):
        self.agent.load(path)