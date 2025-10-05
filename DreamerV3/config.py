import torch, torch.nn as nn


# GENERAL
eps = 1e-12
data_dtype = torch.bfloat16
obs_dtype = torch.uint8 # set NONE to use same as data_dtype
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_name = "SAC-RSSMv3-PENDULUMv1"
save_dir = "saves"

save_path = save_dir + "/" + save_name + "/"
backup_path = save_path + "backup/"
tensorboard_log_dir = "runs/" + save_name + "/"
save_every = 50 # number of episodes
tensorboard_write_every = 128 # every train steps


# ENVIRONMENT
env_name = "Pendulum-v1"
obs_shape = (3, 64, 64) 
act_size = 1 # currently only supports continuous single-head actions
# act_bounds = (-2., 2.)
max_action = 2.0 # Pendulum-v1 specific


### NETWORKS
global_act_fn = nn.SiLU # Could also be ELU, doesn't really matter probably
mlp_layers = 2 * [256,] 
cnn_batchnorm = True

channels = [[32, 64, 128, 256], # conv layers
            [128, 64, 32, 3]]   # deconv layers
kernels = [[4, 4, 4, 4], 
            [5, 5, 6, 6]]
strides = [[2, 2, 2, 2],
            [2, 2, 2, 2]]
paddings = [[0, 0, 0, 0],
            [0, 0, 0, 0]]


### TRAINING
batch_size = 128
lr = 3e-4
gamma = 0.99
alpha = 1e-2
grad_norm_clip = 1.
buffer_size = 2 ** 14
buffer_fill_percentage = 0.1 # how much of buffer to fill before training
train_every = 16 # train every n env steps
loss_stability_scale = 64.


### AGENT
# World Model
h_size = 1024
z_size = 32
s_size = h_size + z_size ** 2 # size of combined state (h, z)
subseq_len = 16

GRU_layers = 1
uniform_mix_ratio = 0.01 # Default 1%

rew_mean_bounds = None # Clamp for stability
rew_logvar_bounds = (-20, 4) # Arbitrary choice, carried over from SAC


# Actor-Critic
raw_mean_range = (-2, 2) # Clamp for stability
act_logvar_bounds = (-20, 4) # Arbitrary choice, carried over from SAC

l2_reg = 0.0 # L2 regularization on actor and critic