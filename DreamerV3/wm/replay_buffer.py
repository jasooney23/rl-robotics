import torch, config as cfg
from torch import Tensor

class ReplayBuffer:
    def __init__(self, capacity, subseq_len, device='cuda'):
        """ capacity: how many subsequences to store 
            subseq_len: length of each subsequence """
        self.capacity = capacity
        self.subseq_len = subseq_len
        self.full = False
        self.device = device
        
        self.idx = 0
        self.buffers = dict(
            h=torch.zeros((capacity, 1, cfg.h_size), dtype=cfg.data_dtype, device=self.device, requires_grad=False),
            obs=torch.zeros((capacity, subseq_len, *cfg.obs_shape), dtype=cfg.obs_dtype, device=self.device, requires_grad=False),
            act=torch.zeros((capacity, subseq_len, cfg.act_size), dtype=cfg.data_dtype, device=self.device, requires_grad=False),
            rew=torch.zeros((capacity, subseq_len, 1), dtype=cfg.data_dtype, device=self.device, requires_grad=False),
            done=torch.zeros((capacity, subseq_len, 1), dtype=cfg.data_dtype, device=self.device, requires_grad=False),
        )

    def add(self, subseq_state_dict: dict):
        """ 
        add a subsequence to the buffer
            
        Args:
            subseq_state_dict: A dictionary of subsequences, one for each signal
        """
        self.buffers["h"][self.idx] = subseq_state_dict["h"][:1] # only store initial h
        self.buffers["act"][self.idx] = subseq_state_dict["act"]
        self.buffers["rew"][self.idx] = subseq_state_dict["rew"]
        self.buffers["done"][self.idx] = subseq_state_dict["done"]

        if cfg.obs_dtype is not None:
            self.buffers["obs"][self.idx] = (subseq_state_dict["obs"] * 255).to(cfg.obs_dtype)
        else:
            self.buffers["obs"][self.idx] = subseq_state_dict["obs"]

        self.idx += 1

        if self.idx >= self.capacity:
            self.idx = 0
            self.full = True

    def sample(self, batch_size, init_step_only=False) -> dict[str, Tensor]:
        '''draw a random minibatch. the buffer does not have to be full to sample from it.
        Args:
            batch_size: number of subsequences to sample
            init_step_only: if true, only return the initial step of each subsequence (for trajectory initialization)
        Returns:
            A dictionary of sampled subsequences, one for each signal. Each tensor is shape (N, T, *) for subsequence length T and batch size N. If init_step_only is true, shape is (N, 1, *).
        '''
        with torch.no_grad():
            if self.full:
                indices = torch.randperm(self.capacity)
            else:
                if batch_size >= self.idx:
                    raise ValueError("screw you")
                indices = torch.randperm(self.idx)
            indices = indices[:batch_size]

            sample = {}
            for k in self.buffers.keys():
                sample[k] = self.buffers[k].index_select(0, indices)[:, :1] if init_step_only else self.buffers[k].index_select(0, indices)

            if cfg.obs_dtype is not None:
                # scale obs back to [0, 1] float
                sample["obs"] = sample["obs"].to(cfg.data_dtype) / 255

        return sample

