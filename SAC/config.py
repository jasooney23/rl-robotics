import torch

''' general settings '''

continuous = True

buffer_min_size = 1e-2 # min percentage of buffer size to start training
buffer_size = int(1e6) # 2 ** 16
buffer_dtype = torch.float32
batch_size = 256 # 128
train_steps_per_update = 1
tensorboard_write_freq = 1000
print_freq = 20
save_freq = 1e4 # 1e4

hidden_layers = 2
layer_size = 256
activation = torch.nn.ELU

eps = 1e-10
gamma = 0.99
lr = 3e-4
l2_reg_q = 0
l2_reg_actor = 0
critic_ema_tau = 5e-3
logstd_range = (-20, 2) # range to clamp log std for numerical stability
raw_mean_range = (-2, 2) # range to clamp raw mean for numerical stability
alpha = 0.05 # default 1e-3 

# ''' SWIMMER '''
# input_shape = 8
# actions = 2
# max_action = 1.0
# tensorboard_log_dir = 'runs/Swimmer_SAC'
# save_path = "saves/Swimmer_SAC/"

# ''' Hopper '''
# input_shape = 11
# actions = 3
# max_action = 1.0
# tensorboard_log_dir = 'runs/Hopper_SAC'
# save_path = "saves/Hopper_SAC/"

''' Humanoid '''
input_shape = 348
actions = 17
max_action = 0.4
tensorboard_log_dir = 'runs/Humanoid_SAC'
save_path = "saves/Humanoid_SAC/"