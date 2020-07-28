from torch import nn
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Bird_DQN(nn.Module):

    def __init__(self, actions_num):
        super(Bird_DQN, self).__init__()

        self.actions_num = actions_num
        self.buildDQN()

    def buildDQN(self):
        self.map_size = (32, 16, 9)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=2).to(device)
        self.relu1 = nn.LeakyReLU(inplace=True).to(device)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1).to(device)
        self.relu2 = nn.LeakyReLU(inplace=True).to(device)

        self.fc1 = nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 128).to(device)
        self.relu3 = nn.LeakyReLU(inplace=True).to(device)
        self.fc2 = nn.Linear(128, self.actions_num).to(device)

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

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
