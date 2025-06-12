'''Soft Actor-Critic (SAC), 2018'''


import torch, numpy as np, config as cfg, gymnasium
from torch import nn as nn
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
np.set_printoptions(precision=2)


import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

from agent import Agent
from trainer import Trainer


'''Setup TensorBoard'''
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(cfg.tensorboard_log_dir)

'''Train code'''

agent = Agent(input_shape=cfg.input_shape, actions=cfg.actions,
              hidden_layers=cfg.hidden_layers, layer_size=cfg.layer_size, activation=cfg.activation)
trainer = Trainer(agent, gamma=cfg.gamma, lr=cfg.lr, act_ent_weight=cfg.act_ent_weight)

writer.add_graph(agent.actor, torch.zeros((1, cfg.input_shape), dtype=torch.float32))

env = gymnasium.make("Ant-v5", render_mode="human")

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
done = False
running_steps = 0
step = 0
time_fin = time_init = 0

while True:
    action, act_mean, act_std = agent.get_action(state)
    next_state, reward, done, truncated, info = env.step(action[0].detach().numpy())

    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    trainer.add_to_buffer(
        state=state, 
        action=action, 
        reward=torch.tensor(reward, dtype=torch.float32),
        next_state=next_state[0],
        done=torch.tensor(done, dtype=torch.float32)
    )
    total_loss, losses = trainer.train()

    state = next_state
    step += 1
    running_steps += 1
    if step % 1 == 0:
        pp = lambda x: x.squeeze().detach().numpy()
        print(f"Step: {step}, Action: {pp(action)}, Mean: {pp(act_mean)}, Std: {pp(act_std)}, Reward: {reward.item():.2f}, Loss: {total_loss:.2f}")

    if done or truncated:
        print(f"Episode finished after {step} steps")
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        step = 0

    if running_steps % cfg.tensorboard_write_freq == 0:
        writer.add_scalar('Losses/total', total_loss, running_steps)
        # writer.add_scalar('Losses/actor', losses['actor'], running_steps)
        # writer.add_scalar('Losses/critic', losses['critic'], running_steps)
        # writer.add_scalar('Losses/actor_ent', losses['actor_ent'], running_steps)

        # for act in range(cfg.actions):
        #     writer.add_scalar(f'Actions/action_{act}/mean', pp(act_mean[:, act]), running_steps)
        #     writer.add_scalar(f'Actions/action_{act}/std', pp(act_std[:, act]), running_steps)
