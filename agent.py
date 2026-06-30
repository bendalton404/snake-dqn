import torch.nn as nn
import torch 
import numpy as np
import copy

class NeuralNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc0 = nn.Linear(in_features=input_size, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=output_size)
        
    # x is a mini batch from the replay buffer
    def forward(self, x):
        x = torch.relu(self.fc0.forward(x))
        x = torch.relu(self.fc1.forward(x))
        return self.fc2.forward(x)


class ConvolutionalNet(nn.Module):
    def __init__(self, input_channels, input_height, input_width, output_size):
        super().__init__()
        self.con0 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=(3,3))
        self.con1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.pool0 = nn.MaxPool2d(kernel_size=(2,2))

        # to find the number of inputs, do a forward pass on zero data 
        # to find the shape of the final convolutional layer
        # e.g. torch.zeros(batch, channels, height, width)
        with torch.no_grad():
            x = torch.zeros((1, input_channels, input_height, input_width))
            x = self.pool0.forward(self.con1.forward(self.con0.forward(x)))
            flatten_size = x.flatten(start_dim=1).shape[1]

        self.fc0 = nn.Linear(in_features=flatten_size, out_features=64)
        self.fc1 = nn.Linear(in_features=64, out_features=output_size)

    # x is a mini batch from the replay buffer
    def forward(self, x):
        x = torch.relu(self.con0.forward(x))
        x = torch.relu(self.con1.forward(x))
        x = self.pool0.forward(x)

        # start_dim=1 so the batch isn't flattened together
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc0.forward(x))
        return self.fc1.forward(x)


class ReplayBuffer:
    def __init__(self, state_shape, buffer_size, min_sample_size):
        self.min_sample_size = min_sample_size
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0

        state_buffer_shape = (buffer_size,) + state_shape
        other_buffer_shape = (buffer_size,)

        self.states = np.zeros(shape=state_buffer_shape, dtype=np.float32)
        self.actions = np.zeros(shape=other_buffer_shape, dtype=np.int64)
        self.rewards = np.zeros(shape=other_buffer_shape, dtype=np.float32)
        self.next_states = np.zeros(shape=state_buffer_shape, dtype=np.float32)
        self.dones = np.zeros(shape=other_buffer_shape, dtype=np.int64)

    def is_ready(self):
        return self.size >= self.min_sample_size

    def add(self, s, a, r, s2, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        states = torch.tensor(self.states[idx], dtype=torch.float32)
        actions = torch.tensor(self.actions[idx], dtype=torch.long)
        rewards = torch.tensor(self.rewards[idx], dtype=torch.float32)
        next_states = torch.tensor(self.next_states[idx], dtype=torch.float32)
        dones = torch.tensor(self.dones[idx], dtype=torch.long)
        return states, actions, rewards, next_states, dones
    

class Agent_MLP:
    def __init__(self):
        # state representation is a 16 element vector
        self.online_net = NeuralNet(input_size=16, output_size=4)
        self.target_net = copy.deepcopy(self.online_net)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(state_dim=(16,), buffer_size=20000, min_sample_size=64)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())


class Agent_CNN:
    def __init__(self):
        # state representation is a 4 channel board
        # channels for walls, apple, snake head, snake body
        # a cell is implicitly grass if it is 0 in all 4 channels
        self.online_net = ConvolutionalNet(input_channels=4, input_height=10, input_width=10, output_size=4)
        self.target_net = copy.deepcopy(self.online_net)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=1e-4)
        self.replay_buffer = ReplayBuffer(state_dim=(4,10,10), buffer_size=20000, min_sample_size=64)
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
