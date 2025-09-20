'''Soft Actor-Critic (SAC), 2018'''
import torch, numpy as np, config as cfg, gymnasium, argparse
from torch import nn as nn

''' set yourself up for success '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH DEVICE: ", device)
torch.set_default_device(device)
np.set_printoptions(precision=2)


''' parse commandline args'''
parser = argparse.ArgumentParser()
parser.add_argument("--nogui", action="store_true", help="Run without GUI")
parser.add_argument("--reset", action="store_true", help="Don't load saved model")
args = parser.parse_args()


''' needed for debugging with torchviz '''
import os, shutil
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
cwd = os.getcwd() + "/"

from agent import Agent
from trainer import Trainer


'''Train code'''
agent = Agent(input_shape=cfg.input_shape, actions=cfg.actions,
              hidden_layers=cfg.hidden_layers, layer_size=cfg.layer_size, activation=cfg.activation)
trainer = Trainer(agent, gamma=cfg.gamma, lr=cfg.lr, alpha=cfg.alpha)

running_steps = 0
ep = 0

train_info = None
if args.reset:
    shutil.rmtree(cwd + cfg.tensorboard_log_dir, ignore_errors=True)
    # old save gets overwritten anyway so no need to delete
    running_steps = 0
    ep = 0
else:
    train_info = trainer.load(cwd + cfg.save_path)
    running_steps = train_info["running_steps"]
    ep = train_info["ep"]
# setup tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(cfg.tensorboard_log_dir)


''' set up environment '''
env = gymnasium.make("Humanoid-v5", render_mode=None if args.nogui else "human")
env.metadata['render_fps'] = 0

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
done = False
ep_return = 0
time_fin = time_init = 0
step = 0


''' Main train loop'''
while True:
    action, act_mean, act_std = agent.get_action(state) # +++
    if action.isnan().any():
        raise ValueError(f"Action contains NaN values: {action}")
    action = action.detach().cpu().numpy()[0]
    
    next_state, reward, done, truncated, info = env.step(action)
    ep_return += reward

    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # +++
    trainer.add_to_buffer( # no net change to memory after
        state=state, 
        action=torch.tensor(action, dtype=torch.float32), 
        reward=torch.tensor(reward, dtype=torch.float32), #  # +++, this does not get released
        next_state=next_state[0],
        done=torch.tensor(done, dtype=torch.float32) # This DOES get released
    )
    total_loss, metrics = trainer.train()

    state = next_state
    step += 1
    running_steps += 1
    if step % cfg.print_freq == 0:
        pp = lambda x: x.squeeze().detach().cpu().numpy()
        if cfg.continuous:
            print(f"Step: {step}, Action: {action}, Mean: {pp(act_mean)}, Std: {pp(act_std)}, Reward: {reward:.2f}, Loss: {total_loss:.2f}, Loss_Act: {metrics['actor']:.2f}, Loss_Critic: {metrics['critic']:.2f} Avg std: {metrics['avg_std']:.2f}, Avg mean: {metrics['avg_mean']:.2f}, Max std: {metrics['max_std']:.2f}, Max abs mean: {metrics['max_abs_mean']:.2f}, ")
            pass
        else:
            # target_critic_val = agent.target_critic((state, nn.functional.one_hot(action, num_classes=cfg.actions).float()))
            # print(f"Step: {step}, Action_probs: {pp(act_mean)}, Critic: {pp(critic_val):.4f}, Advantage: {pp(reward - (cfg.gamma * (state, nn.functional.one_hot(action, num_classes=cfg.actions).float()) * (1-done)))}, Reward: {reward:.2f}, Loss: {total_loss:.2f}")
            pass
    if done or truncated:
        print(f"Episode finished after {step} steps")
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        step = 0
        writer.add_scalar("Return per episode", ep_return, ep)
        ep += 1
        ep_return = 0

    if running_steps % cfg.tensorboard_write_freq == 0:
        writer.add_scalar('Losses/total', total_loss, running_steps)
        writer.add_scalar('Losses/actor', metrics['actor'], running_steps)
        writer.add_scalar('Losses/critic', metrics['critic'], running_steps)
        writer.add_scalar('Losses/actor_ent', metrics['actor_ent'], running_steps)
        writer.add_scalar('Metrics/avg_std', metrics['avg_std'], running_steps)
        writer.add_scalar('Metrics/avg_mean', metrics['avg_mean'], running_steps)
        writer.add_scalar('Metrics/max_std', metrics['max_std'], running_steps)
        writer.add_scalar('Metrics/max_abs_mean', metrics['max_abs_mean'], running_steps)

    if (running_steps + 1) % cfg.save_freq == 0:
        train_info = {
            "running_steps": running_steps,
            "ep": ep,
        }
        trainer.save(cwd + cfg.save_path, train_info)
        trainer.save(cwd + cfg.save_path + "backup/", train_info)
        print(f"Model saved at step {running_steps}")
