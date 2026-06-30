import torch.nn as nn
import torch 
import numpy as np
import copy

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc0 = nn.Linear(input_size, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc0.forward(x))
        x = torch.relu(self.fc1.forward(x))
        return self.fc2.forward(x)
 

class ReplayBuffer:
    def __init__(self, state_dim, max_size, min_sample_size):
        self.min_sample_size = min_sample_size
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size,), dtype=np.int64)
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size,), dtype=np.int64)

    def is_ready(self):
        return self.size >= self.min_sample_size

    def add(self, s, a, r, s2, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)

        states = torch.tensor(self.states[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.long)
        rewards = torch.tensor(self.rewards[idx], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[idx], dtype=torch.float32)
        dones = torch.tensor(self.dones[idx], dtype=torch.long)

        return states, actions, rewards, next_states, dones
    

class Agent:
    def __init__(self):
        # state representation is a 16 element vector for snake
        # 4 actions possible, go NESW
        self.online_net = NeuralNet(16, 4)
        self.target_net = copy.deepcopy(self.online_net)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(16, 20000, 64)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())