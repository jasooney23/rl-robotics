'''Soft Actor-Critic (SAC), 2018'''
import torch, numpy as np, config as cfg, gymnasium, os, time
from torch import nn as nn
from agent import Agent
from trainer import Trainer


''' set yourself up for success '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PYTORCH DEVICE: ", device)
torch.set_default_device(device)


'''Train code'''
agent = Agent(input_shape=cfg.input_shape, actions=cfg.actions,
              hidden_layers=cfg.hidden_layers, layer_size=cfg.layer_size, activation=cfg.activation)
trainer = Trainer(agent, gamma=cfg.gamma, lr=cfg.lr, alpha=cfg.alpha)
cwd = os.getcwd() + "/"
trainer.load(cwd + cfg.save_path)


''' set up environment '''
env = gymnasium.make("Humanoid-v5", render_mode="human")
state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
done = False
FPS = 40

''' Main train loop'''
while True:
    start_time = time.time()
    action, act_mean, act_std = agent.get_action(state)
    action = action.detach().cpu().numpy()[0]
    next_state, reward, done, truncated, info = env.step(action)

    next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
    state = next_state
    fin_time = time.time()
    if fin_time - start_time < 1.0 / FPS:
        time.sleep(1.0 / FPS - (fin_time - start_time))

    if done or truncated:
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)