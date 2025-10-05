import torch

class Agent:
    def __init__(self, input_shape, actions: int):
        self.input_shape = input_shape
        self.actions = actions


    def get_action(self, state: torch.Tensor, *args, **kwargs):
        raise NotImplementedError
    

    def set_eval(self):
        if hasattr(self, 'nets'):
            [net.eval() for net in self.nets]
        else:
            print("No nets to set to eval mode")


    def set_train(self):
        if hasattr(self, 'nets'):
            [net.train() for net in self.nets]
        else:
            print("No nets to set to train mode")

    
    def save(self, path: str):
        raise NotImplementedError
    

    def load(self, path: str):
        raise NotImplementedError
    

class AgentTrainer:
    def __init__(self, agent: Agent):
        self.agent = agent


    def train_step(self, batch: dict):
        raise NotImplementedError
