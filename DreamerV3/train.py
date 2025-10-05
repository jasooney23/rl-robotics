'''Soft Actor-Critic (SAC), 2018'''
import torch, numpy as np, config as cfg, gymnasium, argparse
from torch import nn as nn
from common.utils import bcall, itemify, numpify

''' set yourself up for success '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH DEVICE: ", device)
torch.set_default_device(device)
np.set_printoptions(precision=2)
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True


''' parse commandline args'''
parser = argparse.ArgumentParser()
parser.add_argument("--nogui", action="store_true", help="Run without GUI")
parser.add_argument("--reset", action="store_true", help="Don't load saved model")
args = parser.parse_args()


''' needed for debugging with torchviz '''
import os, shutil
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
cwd = os.getcwd() + "/"

from agent.BaseSAC import BaseSAC
from agent.BaseSAC import BaseSAC_Trainer
from wm.RSSM import RSSM


''' basic metrics & logging '''
train_info = None
running_steps = 0
ep = 0
save_path = cwd + cfg.save_path
backup_path = cwd + cfg.backup_path

if args.reset:
    shutil.rmtree(cwd + cfg.tensorboard_log_dir, ignore_errors=True)
    # old save gets overwritten anyway so no need to delete
    running_steps = 0
    ep = 0    
    # make save dirs if they don't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + "agent/"):
        os.makedirs(save_path + "agent/")
    if not os.path.exists(save_path + "world_model/"):
        os.makedirs(save_path + "world_model/")
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    if not os.path.exists(backup_path + "agent/"):
        os.makedirs(backup_path + "agent/")
    if not os.path.exists(backup_path + "world_model/"):
        os.makedirs(backup_path + "world_model/")
else:
    # load saved model if it exists
    train_info = torch.load(save_path + "train_info.pt")
    running_steps = train_info["running_steps"]
    ep = train_info["ep"]
# setup tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(cwd + cfg.tensorboard_log_dir)


""" util functions """
itemify_dict = lambda d: {k: itemify(v) for k, v in d.items()}
def process_state(state, resize=True):
    ''' process state from env.render()'''
    from PIL import Image
    state = Image.fromarray(state)
    if resize:
        state = state.resize(cfg.obs_shape[1:], Image.BILINEAR) # (H, W)
    state = np.array(state)
    state = torch.tensor(state, dtype=cfg.data_dtype, requires_grad=False).permute(2, 0, 1) / 255 # (C, H, W)
    return state
def save(agent_trainer, world_model, train_info, path):
    agent_trainer.save(path + "agent/")
    world_model.save(path + "world_model/")
    torch.save(train_info, path + "train_info.pt")


''' Set up components'''
agent = BaseSAC(
    input_shape=cfg.s_size, 
    actions=cfg.act_size,
)
world_model = RSSM(
    obs_shape=cfg.obs_shape, 
    h_size=cfg.h_size, 
    z_size=cfg.z_size, 
    act_size=cfg.act_size,
    lr=cfg.lr,
    clip_grad=cfg.grad_norm_clip,
    load=not args.reset,
    load_path=cwd + cfg.save_path + "rssm/",
)
agent_trainer = BaseSAC_Trainer(
    agent=agent,
    world_model=world_model,
    gamma=cfg.gamma, 
    lr=cfg.lr, 
    alpha=cfg.alpha,
    batch_size=cfg.batch_size,
    subseq_len=cfg.subseq_len,
)
if not args.reset: agent_trainer.load(save_path)


''' set up environment '''
env = gymnasium.make(cfg.env_name, render_mode="rgb_array")

_, info = env.reset()
real_state = process_state(env.render()) # initial state
done = False
ep_return = 0
time_fin = time_init = 0
step = 0

agent_loss = wm_loss = np.nan
losses = metrics = {}

''' Main train loop'''
while True:
    """ Take action in environment """
    latent_state = world_model.get_latent_state(real_state, combined=True)
    action = [itemify(agent.get_step_action(latent_state)),] # sample from policy
    _, reward, done, truncated, info = env.step(action) # step
    next_real_state = env.render()
    ep_return += reward

    """ Update RSSM with transition """
    state_transition = dict(
        obs=real_state,
        state=latent_state,
        act=torch.tensor(action, dtype=cfg.data_dtype, requires_grad=False) / cfg.max_action, # scale action to [-1, 1]
        rew=torch.tensor((reward,), dtype=cfg.data_dtype, requires_grad=False),
        done=torch.tensor((done,), dtype=cfg.data_dtype, requires_grad=False),
    )
    world_model.update_state(state_transition, push_buffer=True) # add transition to subsequence buffer & push to replay buffer (if full)

    """ Train world model & agent """
    train_this_step = (world_model.buffer.idx >= cfg.buffer_size * cfg.buffer_fill_percentage or world_model.buffer.full) \
                        and running_steps % cfg.train_every == 0
    if train_this_step:
        # train if buffer is sufficiently full and every _ steps
        agent_loss, agent_losses, agent_metrics = agent_trainer.train_on_subseq()
        wm_loss, wm_losses, wm_metrics = world_model.learn_on_batch()

        losses = {**agent_losses, **wm_losses}; metrics = {**agent_metrics, **wm_metrics}
        losses = itemify_dict(losses); metrics = itemify_dict(metrics)

        print(f"Step: {step}, Action: {action}, Reward: {reward:.2f}, Agent Loss: {itemify(agent_loss):.2f}, WM Loss: {itemify(wm_loss):.2f}")

    """ Next step prep"""
    real_state = process_state(next_real_state)
    step += 1
    running_steps += 1

    if done or truncated:
        print(f"Episode finished after {step} steps")
        _, info = env.reset()
        real_state = process_state(env.render())
        world_model.reset_subseqs() # reset RSSM state

        writer.add_scalar("Return per episode", ep_return, ep)
        if (ep + 1) % cfg.save_every == 0:
            train_info = { "running_steps": running_steps, "ep": ep, }
            save(agent_trainer, world_model, train_info, save_path)
            save(agent_trainer, world_model, train_info, backup_path)

            print(f"MODEL SAVED AT {running_steps} TOTAL STEPS")

        step = 0
        ep += 1
        ep_return = 0


    """ Logging & saving """
    if running_steps % cfg.tensorboard_write_every == 0:
        writer.add_scalar('Losses/AGENT_TOTAL', agent_loss, running_steps)
        writer.add_scalar('Losses/WORLDMODEL_TOTAL', wm_loss, running_steps)

        for k, v in losses.items(): # all losses & metrics displayed together (agent/wm)
            writer.add_scalar(f'Losses/{k}', v, running_steps)
        for k, v in metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, running_steps)


