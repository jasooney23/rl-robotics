import torch

continuous = True

buffer_min_size = 0.125 # min percentage of buffer size to start training
buffer_size = int(1e6) # 2 ** 16
buffer_dtype = torch.float32
batch_size = 256 # 128
train_steps_per_update = 1

eps = 1e-10
gamma = 0.99
lr = 3e-4
l2_reg_q = 0
l2_reg_actor = 0
critic_ema_tau = 5e-3
gradnorm_clip = 1e5

hidden_layers = 2
layer_size = 256
activation = torch.nn.ReLU

input_shape = 8
actions = 2
max_action = 1.0
max_std = 4.0

alpha = 5e-1 # default 1e-3 

tensorboard_log_dir = 'runs/Swimmer_SAC'
tensorboard_write_freq = 100
print_freq = 20
save_path = "saves/Swimmer_SAC/"
backup_path = "saves/Swimmer_SAC_backup/"
save_freq = 1e4 # 1e4