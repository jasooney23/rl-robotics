import torch

continuous = True

buffer_min_size = 0.125 # min percentage of buffer size to start training
buffer_size = 2**14
buffer_dtype = torch.float32
batch_size = 256
train_steps_per_update = 1

eps = 1e-8
gamma = 0.99
lr = 3e-4
critic_ema_tau = 5e-3
gradnorm_clip = 1

hidden_layers = 2
layer_size = 256
activation = torch.nn.ELU

input_shape = 9
actions = 1
clip = (-1, 1)  # action clipping rangem
max_action = 1.0
max_std = 4.0

alpha = 1e-1 # default 1e-3

tensorboard_log_dir = 'runs/InvDP_SAC'
tensorboard_write_freq = 100
print_freq = 10
save_path = "saves/InvDP_SAC/"
backup_path = "saves/InvDP_SAC_backup/"
save_freq = 1e4