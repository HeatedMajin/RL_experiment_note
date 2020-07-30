import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical

device = torch.device("cpu")  # "cuda" if torch.cuda.is_available() else "cpu")


class Dense_Policy(nn.Module):
    def __init__(self, input_features, action_nums):
        super(Dense_Policy, self).__init__()

        self.gamma = 0.95
        self.eps = np.finfo(np.float64).eps.item()
        self.train_step = False

        self.build_layers(input_features, action_nums)

        ######### 每个被选取过的action保存 #############
        # 这个action的log prob和这个action的reward
        self.log_probs_action = []
        self.reward_pre_action = []
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def build_layers(self, input_features, action_nums):
        self.dense1 = nn.Linear(input_features, input_features * 32).to(device)
        self.drop1 = nn.Dropout(p=0.4).to(device)
        self.dense2 = nn.Linear(input_features * 32, action_nums * 8).to(device)
        self.drop2 = nn.Dropout(p=0.4).to(device)
        self.dense3 = nn.Linear(action_nums * 8, action_nums).to(device)

    def forward(self, obs):
        res = F.relu(self.drop1(self.dense1(obs)))
        res = F.relu(self.drop2(self.dense2(res)))
        action_score = self.dense3(res)
        return F.softmax(action_score, dim=1)

    def select_action(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action_probs = self.forward(state)
        m = Categorical(action_probs)  # action的概率分布
        action = m.sample()  # sample出一个action

        if self.train_step:
            ##### 保存选取的这个action的log prob ####
            self.log_probs_action.append(m.log_prob(action))

        return action.item()

    def update_policy(self):
        R = 0
        policy_loss, returns = [], []
        for r in self.reward_pre_action[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(self.log_probs_action, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        print(policy_loss.item())
        policy_loss.backward()
        self.optimizer.step()

        del self.log_probs_action[:]
        del self.reward_pre_action[:]

    def save_policy(self, save_path, save_filename):
        torch.save(self.state_dict(), os.path.join(save_path, save_filename))
