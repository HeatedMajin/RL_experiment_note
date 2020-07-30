import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.distributions import Categorical

device = torch.device("cpu")  # cuda" if torch.cuda.is_available() else "cpu")


class DCN_policy(nn.Module):
    def __init__(self, actions_nums):
        super(DCN_policy, self).__init__()

        self.actions_num = actions_nums
        self.buildDCN()

        self.train_step = False
        self.gamma = 0.99

        self.saved_log_probs = []
        self.rewards = []
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-3)#, weight_decay=0.99)

    def buildDCN(self):
        # self.map_size = (16, 10, 10)
        #
        # self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=3, padding=2).to(device)
        # self.relu1 = nn.LeakyReLU().to(device)
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1).to(device)
        # self.relu2 = nn.LeakyReLU().to(device)
        #
        # self.fc1 = nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 256).to(device)
        # self.relu3 = nn.LeakyReLU().to(device)
        # self.fc2 = nn.Linear(256, self.actions_num).to(device)

        self.map_size = (8, 10, 10)

        self.conv1 = nn.Conv2d(1, 8, kernel_size=10, stride=6, padding=2).to(device)
        self.relu1 = nn.LeakyReLU().to(device)

        self.fc1 = nn.Linear(self.map_size[0] * self.map_size[1] * self.map_size[2], 64).to(device)
        self.relu3 = nn.LeakyReLU().to(device)
        self.fc2 = nn.Linear(64, self.actions_num).to(device)

    def forward(self, state):
        out = self.conv1(state)
        out = self.relu1(out)
        # out = self.conv2(out)
        # out = self.relu2(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.relu3(out)
        out = self.fc2(out)

        return F.softmax(out, dim=1)

    def take_action(self, obs):
        # obs = Variable([obs]).unsqueeze(dim=0).to(device)
        obs = Variable(torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0)).to(device)
        probs = self.forward(obs)

        # 输入一个分布，sample出分布上的点
        m = Categorical(probs)
        action = m.sample()

        if self.train_step:
            # 记录action的分布（后面计算loss）
            self.saved_log_probs.append(m.log_prob(action))

        return action.item()



    def update_policy(self):
        R = 0
        policy_loss,returns = [],[]
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        # turn rewards to pytorch tensor and standardize
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        for log_prob, reward in zip(self.saved_log_probs, returns):
            policy_loss.append(- log_prob * reward)

        self.optimizer.zero_grad()

        policy_loss = torch.stack(policy_loss).sum()
        print("\t"+str(policy_loss.item()))
        policy_loss.backward()
        self.optimizer.step()

        # clean rewards and saved_actions
        del self.rewards[:]
        del self.saved_log_probs[:]
