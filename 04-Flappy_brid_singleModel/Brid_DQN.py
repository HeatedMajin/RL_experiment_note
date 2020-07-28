from collections import deque

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bird_DQN(nn.Module):
    ''''
    Deep Q network for flapping bird:
        Given image , the network will output what the action should be taken
    '''

    empty_frame = np.zeros((72, 128), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame), axis=0)

    def __init__(self, epsilon, mem_size):
        super(Bird_DQN, self).__init__()

        self.epsilon = epsilon

        self.actions_num = 2
        self.buildDQN()

        self.trainable = None

        self.mem_size = mem_size
        self.replay_mem = deque()

        self.time_step = 0

    def buildDQN(self):
        self.map_size = (32, 16, 9)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=2).to(device)
        self.relu1 = nn.LeakyReLU(inplace=True).to(device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1).to(device)
        self.relu2 = nn.LeakyReLU(inplace=True).to(device)

        self.fc1 = nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 128).to(device)
        self.relu3 = nn.LeakyReLU(inplace=True).to(device)
        self.fc2 = nn.Linear(128, self.actions_num).to(device)

    def set_trainable(self, trainable):
        self.trainable = trainable

    def set_initial_state(self, obs=None):
        """
        Set initial state
        state: initial state. if None, use `BrainDQN.empty_state`
        """
        if obs is None:
            self.current_state = Bird_DQN.empty_state

        else:
            self.current_state = np.append(Bird_DQN.empty_state[1:, :, :], obs.reshape((1,) + obs.shape),
                                           axis=0)


    def forward(self, obs):
        # get Q estimation
        out = self.conv1(obs)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)
        return out

    def optimal_action(self):  # greedy choose (exploitation)
        state = self.current_state
        state_var = Variable(torch.from_numpy(state), requires_grad=False).unsqueeze(0).to(device)

        q_values = self.forward(state_var)
        _, actions_index = torch.max(q_values, dim=1)
        action_index = actions_index.data[0]
        action = np.zeros(self.actions_num, dtype=np.float32)
        action[action_index] = 1
        return action

    def random_action(self):  # random choose (exploration)
        action = np.zeros(self.actions_num, dtype=np.float32)
        action_index = 0 if np.random.random() < 0.8 else 1
        action[action_index] = 1
        return action

    def take_action(self):
        if np.random.random() < self.epsilon:
            return self.random_action()
        else:
            return self.optimal_action()

    def store_trans(self, action, reward, next_obs, finish):
        next_state = np.append(self.current_state[1:, :, :], next_obs.reshape((1,) + next_obs.shape), axis=0)
        self.replay_mem.append((self.current_state, action, reward, next_state, finish))

        if (len(self.replay_mem) > self.mem_size):
            self.replay_mem.popleft()

        if (finish):
            self.set_initial_state()
        else:
            self.current_state = next_state

    def increase_timestep(self):
        self.time_step += 1
