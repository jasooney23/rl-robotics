import torch, numpy as np, torch.nn as nn, torch.nn.functional as F, time, os
from typing import List, Union
from torch import Tensor

import common.utils as utils, config as cfg
from common.utils import bcall

from wm.replay_buffer import ReplayBuffer
from wm.models import SequentialModel, LatentEncoder, LatentDecoder, RewardPredictor, DynamicPredictor, TerminationPredictor
from wm.replay_buffer import ReplayBuffer

class RSSM:
    def __init__(self, 
                 obs_shape: List[int]=cfg.obs_shape,
                 gru_layers: int=cfg.GRU_layers,
                 mlp_layers: List[int]=cfg.mlp_layers,
                 h_size: int=cfg.h_size, 
                 z_size: int=cfg.z_size,
                 act_size: int=cfg.act_size, 
                 lr: float=0.001, 
                 L_pred_weight: float=1.0,
                 L_dyn_weight: float=1.0,
                 L_rep_weight: float=0.1,
                 clip_grad: float=None,
                 load: bool=False, 
                 load_path: str=None):
        
        # core models for RSSM
        self.seq: SequentialModel = SequentialModel(h_size, z_size, act_size, gru_layers=gru_layers).to(cfg.device).to(cfg.data_dtype)
        self.enc: LatentEncoder = LatentEncoder(h_size, z_size, obs_shape, batchnorm=cfg.cnn_batchnorm).to(cfg.device).to(cfg.data_dtype)
        self.dyn: DynamicPredictor = DynamicPredictor(h_size, z_size, mlp_layers=mlp_layers).to(cfg.device).to(cfg.data_dtype)
        self.rew: RewardPredictor = RewardPredictor(h_size, z_size, mlp_layers=mlp_layers).to(cfg.device).to(cfg.data_dtype)
        self.don: TerminationPredictor = TerminationPredictor(h_size, z_size, mlp_layers=mlp_layers).to(cfg.device).to(cfg.data_dtype)
        self.dec: LatentDecoder = LatentDecoder(h_size, z_size, batchnorm=cfg.cnn_batchnorm, mlp_layers=mlp_layers).to(cfg.device).to(cfg.data_dtype)
        self.nets = [self.seq, self.enc, self.dyn, self.rew, self.don, self.dec]

        self.h_size = h_size
        self.z_size = z_size
        self.act_size = act_size

        self.clip_grad = clip_grad
        self.loss_weights = dict(
            pred=L_pred_weight,
            dyn=L_dyn_weight,
            rep=L_rep_weight
        )

        if load:
            print("Checking for saved model @ ", load_path)
            try:
                print("\n\n")
                self.load(load_path)
                print("\n\n")
            except Exception as e:
                raise RuntimeError(f"Could not load model from {load_path}. Error: \n{e}")
        else:
            self.buffer = ReplayBuffer(capacity=cfg.buffer_size, subseq_len=cfg.subseq_len)

        # optimizer & learning stuff
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.params = (list(self.seq.parameters()) 
                    + list(self.enc.parameters())
                    + list(self.dyn.parameters())
                    + list(self.don.parameters())
                    + list(self.rew.parameters())
                    + list(self.dec.parameters()))
        self.optim = torch.optim.Adam(self.params, lr=lr)

        # keep subsequences of past states to pass to the buffer
        self.reset_subseqs()


    def get_internal_state(self, numpy: bool=False, batchify: bool=False) -> dict:
        """ 
        return the current state by extracting from the last index of subsequence buffer queue 
        
        Args:
            numpy: Whether to return the state as a NumPy array.
            batchify: Whether to add a batch dimension to the returned tensors. (unqueeze dim 0)
        
        Returns:
            dict: The current state.
        """
        state = {}
        for k, v in self.subseqs.items():
            state[k] = v[self.subseq_index]
            if numpy:
                state[k] = state[k].cpu().numpy()
            if batchify:
                state[k] = state[k].unsqueeze(0)

        return state


    def get_latent_state(self, obs, combined: bool=True) -> Union[Tensor, tuple[Tensor, Tensor]]:
        '''returns the next latent state, given the current subsequence buffer. Singular (unbatched) inputs only.
        Args:
            obs: The current observation from the environment. Should be normalized.
            combined: Return as s = cat((h, z)) or (h, z) tuple
        Returns:
            If combined, returns s = cat((h, z))
            Else, returns (h, z) tuple
        '''
        self.set_eval()
        with torch.no_grad():
            prev_state = self.get_internal_state()
            h = bcall(self.seq, prev_state["h"], prev_state["z"], prev_state["act"])
            z = bcall(self.enc, h, obs)
            if combined:
                return torch.cat((h, z), dim=-1)
            return h, z


    def update_state(self, state_dict: dict[str, Tensor], push_buffer=False):
        """ 
        push a new tuple of (s, x, a, r, c) onto the subsequence buffer 
            
        Args:
            state: Latent state s = cat((h, z))
            obs: The current observation from the environment. Should be normalized.
            act: The action taken by the agent. Should be normalized (in [-1, 1]).
            rew: The reward received from the environment.
            done: The termination signal (1 for terminal state).

        Returns:
            None
        """
        state = state_dict["state"]
        obs = state_dict["obs"]
        act = state_dict["act"]
        rew = state_dict["rew"]
        done = state_dict["done"]

        idx = self.subseq_index
        self.subseqs["h"][idx] = state[:self.h_size]
        self.subseqs["z"][idx] = state[self.h_size:]
        self.subseqs["obs"][idx] = obs
        self.subseqs["act"][idx] = act
        self.subseqs["rew"][idx] = rew
        self.subseqs["done"][idx] = done

        if idx < cfg.subseq_len - 1:
            # not full yet
            self.subseq_index += 1
        elif idx == cfg.subseq_len - 1:
            # full
            if push_buffer:
                self.buffer.add(self.subseqs)
            for k in self.subseqs.keys(): # shift to the left, to make space for latest
                self.subseqs[k] = torch.cat((self.subseqs[k][1:], self.subseqs[k][:1]))


    def reset_subseqs(self):
        """ resets the subsequence buffer to ALL ZEROES """
        self.subseq_index = 0
        self.subseqs = dict(
            h=torch.zeros((cfg.subseq_len, cfg.h_size), dtype=cfg.data_dtype, requires_grad=False,),
            z=torch.zeros((cfg.subseq_len, cfg.z_size ** 2), dtype=cfg.data_dtype, requires_grad=False,),
            obs=torch.zeros((cfg.subseq_len, *cfg.obs_shape), dtype=cfg.data_dtype, requires_grad=False,),
            act=torch.zeros((cfg.subseq_len, cfg.act_size), dtype=cfg.data_dtype, requires_grad=False,),
            rew=torch.zeros((cfg.subseq_len, 1), dtype=cfg.data_dtype, requires_grad=False,),
            done=torch.zeros((cfg.subseq_len, 1), dtype=cfg.data_dtype, requires_grad=False,),
        )


    def loss_step(self, state: dict) -> tuple[Tensor, dict, dict, dict]:
        """ computes loss function for a single time step
        Args:
            state: dict of Tensors with keys h, z, x, r, d
            h: (N, h_size) - recurrent latent state
            z: (N, z_size ** 2) - stochastic latent state
            obs: (N, obs_shape) - observation
            rew: (N, 1) - reward
            done: (N, 1) - done signal
        Returns:
            loss: (Tensor) - Combined weighted summed losses
            next_state: (dict) - next state in rollout
            losses: (dict) - individual losses
            metrics: (dict) - various metrics for logging
        """

        h = state["h"]; z = state["z"]; x = state["obs"]; r = state["rew"]; d = state["done"]

        pred_dec = self.dec(h, z)
        pred_rew = self.rew(h, z)
        pred_don = self.don(h, z)
        L_pred = torch.mean((pred_dec - x) ** 2, dim=(-3, -2, -1)).unsqueeze(-1) + (pred_rew - r) ** 2 + (pred_don - d) ** 2

        pred_enc, probs_enc = self.enc(h, x, return_probs=True)
        _, probs_dyn = self.dyn(h, return_probs=True)
        L_dyn = torch.clamp(utils.softmax_kldiv(probs_enc.detach(), probs_dyn), min=1.)
        L_rep = torch.clamp(utils.softmax_kldiv(probs_enc, probs_dyn.detach()), min=1.)

        loss = (L_pred.mean() * self.loss_weights["pred"] +
                L_dyn.mean() * self.loss_weights["dyn"] +
                L_rep.mean() * self.loss_weights["rep"])
        
        losses = dict(RSSM_L_pred=L_pred.mean().item(),
                      RSSM_L_dyn=L_dyn.mean().item(),
                      RSSM_L_rep=L_rep.mean().item())
        
        metrics = {}

        # return zt so next step can use for ht+1
        next_state = dict(
            z=pred_enc,
        )

        return loss, next_state, losses, metrics


    def loss_over_subseq(self, batch: dict) -> tuple[Tensor, dict, dict]:
        """ computes loss function over a batch of subsequences from time 1:T
        Args:
            batch: dict of Tensors with keys h, act, obs, rew, done
            h: (N, 1, h_size) - recurrent latent state at time 0
            act: (N, T, act_size) - action subseq
            obs: (N, T, *obs_shape) - observation subseq
            rew: (N, T, 1) - reward subseq
            done: (N, T, 1) - done signal subseq
        Returns:
            loss_total: (Tensor) - Combined weighted summed losses
            losses: (dict) - individual losses
            metrics: (dict) - various metrics for logging
        """

        h_batch = batch["h"]; act_batch = batch["act"]; obs_batch = batch["obs"]; rew_batch = batch["rew"]; done_batch = batch["done"]

        total_losses = []
        losses = []; metrics = []
        N = obs_batch.shape[0]
        T = obs_batch.shape[1]

        # Loop through each time step in the subsequence, summing the loss.
        for t in range(T):
            xt = obs_batch[:, t]; at = act_batch[:, t]; rt = rew_batch[:, t]; dt = done_batch[:, t]

            if t == 0:
                ht = h_batch[:, t]
                zt = self.enc(ht, xt)
            else:
                ht = self.seq(ht, zt.detach(), at)

            state = dict(h=ht, z=zt, obs=xt, rew=rt, done=dt)
            loss_step, next_state, losses_step, metrics_step = self.loss_step(state)
            zt = next_state["z"]

            total_losses.append(loss_step)
            losses.append(losses_step)
            metrics.append(metrics_step)

        # Aggregate losses and metrics
        loss_total = torch.stack(total_losses).mean()
        # losses = self.agg_dict(losses)
        # metrics = self.agg_dict(metrics)
        losses = {}
        metrics = {}

        return loss_total, losses, metrics


    def learn_on_batch(self, batch: dict[str, Tensor]=None) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        """subsequence: time t from 1 to T (subseq length).
            offline."""
        self.set_train()

        if batch is None:
            batch = self.buffer.sample(batch_size=cfg.batch_size)

        loss_total, losses, metrics = self.loss_over_subseq(batch)

        loss_total = loss_total * cfg.loss_stability_scale
        self.optim.zero_grad()
        loss_total.backward()
        for p in self.params: # scale up loss since bfloat16 has lower decimal precision, then scale back down gradients to be consistent
            if p.grad is not None:
                p.grad.data = p.grad.data / cfg.loss_stability_scale
        '''clip gradients. needs to be fairly small to be stable.'''
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.optim.step()

        loss_total = loss_total.detach() / cfg.loss_stability_scale

        return loss_total, losses, metrics


    def imagine_trajectory(self, h0: Tensor, s0: Tensor, action: Tensor, length=cfg.subseq_len):
        """Imagines a trajectory of latent states using dyn().
           util function to help with debugging."""
        
        raise NotImplementedError("i need to fix this still")

        with torch.no_grad(): # "memory leak" if not used
            traj = dict(h=torch.zeros((length, cfg.deter_size), dtype=torch.float32),
                    s=torch.zeros((length, cfg.stoch_size, cfg.stoch_size), dtype=torch.float32).softmax(dim=1))
            
            traj["h"][0] = h_init
            traj["s"][0] = s_init
            '''again, didn't figure out how to parallelize this, so it's done sequentially.'''
            for i in range(length - 1): # Loop through each time step, imagine next state with previous
                s_sample = utils.softmax_sample.apply(traj["s"][i].unsqueeze(0))
                traj["h"][i + 1] = self.deter(traj["h"][i].unsqueeze(0), s_sample, action.reshape(1, -1))
                traj["s"][i + 1] = self.dyn(traj["h"][i + 1].unsqueeze(0))

            SAMPLE = utils.softmax_sample.apply(traj["s"])
            recs = self.rec(traj["h"], SAMPLE)

            return traj, recs
        
    
    def agg_dict(self, dicts: List[dict]) -> dict:
        '''aggregates a list of dicts by averaging the values for each key'''
        agg = {}
        for k in dicts[0].keys():
            agg[k] = sum(d[k] for d in dicts) / len(dicts)
        return agg
        
        
    def save(self, path):
        print("RSSM: Saving in 2 seconds")
        time.sleep(2)
        print("RSSM: Saving...", end="\r")
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.seq.state_dict(), path + "seq.pt")
        torch.save(self.dyn.state_dict(), path + "dyn.pt")
        torch.save(self.enc.state_dict(), path + "enc.pt")
        torch.save(self.dec.state_dict(), path + "dec.pt")
        torch.save(self.rew.state_dict(), path + "rew.pt")
        torch.save(self.don.state_dict(), path + "don.pt")
        torch.save(self.buffer, path + "replay_buffer.pt")

        print(f"RSSM: Saved model weights & buffer to {path}", end="\r")


    def load(self, path):
        self.seq.load_state_dict(torch.load(path + "seq.pt"))
        self.dyn.load_state_dict(torch.load(path + "dyn.pt"))
        self.enc.load_state_dict(torch.load(path + "enc.pt"))
        self.dec.load_state_dict(torch.load(path + "dec.pt"))
        self.rew.load_state_dict(torch.load(path + "rew.pt"))
        self.don.load_state_dict(torch.load(path + "don.pt"))
        self.buffer = torch.load(path + "replay_buffer.pt")

        print(f"RSSM: Loaded model weights & buffer from {path}")


    def set_eval(self):
        for net in self.nets:
            net.eval()


    def set_train(self):
        for net in self.nets:
            net.train()