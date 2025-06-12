import torch

buffer_min_size = 0.250 # min percentage of buffer size to start training
buffer_size = 2**14
buffer_dtype = torch.float32
batch_size = 32
train_steps_per_update = 4

eps = 1e-8
gamma = 0.99
lr = 1e-4
critic_ema_tau = 1e-3

hidden_layers = 3
layer_size = 128
activation = torch.nn.ELU

input_shape = 105
actions = 8
clip = (-1, 1)  # action clipping range
max_std = 4.0

act_ent_weight = 1e-3

tensorboard_log_dir = 'runs/a2c_ant_v5'
tensorboard_write_freq = 100