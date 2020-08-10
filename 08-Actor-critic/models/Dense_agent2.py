import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical

device = torch.device("cpu")  #"cuda" if torch.cuda.is_available() else "cpu")


class Dense_Policy(nn.Module):
    def __init__(self, input_features, action_nums, gamma):
        super(Dense_Policy, self).__init__()

        self.gamma = gamma
        self.eps = np.finfo(np.float64).eps.item()

        self.build_layers(input_features, action_nums)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.lr = 1e-3

        # 这个action的log prob和这个action的reward
        self.logProbs_values = []
        self.reward_batch = []

        self.policy_grads = {}
        self.policy_params = {}
        for name, p in self.named_parameters():
            if "dense" in name or "action" in name:
                self.policy_params[name] = p

    def build_layers(self, input_features, action_nums):
        self.dense1 = nn.Linear(input_features, input_features * 16).to(device)
        self.drop1 = nn.Dropout(p=0.2).to(device)
        self.dense2 = nn.Linear(input_features * 16, action_nums * 32).to(device)
        self.drop2 = nn.Dropout(p=0.2).to(device)
        self.dense3 = nn.Linear(action_nums * 32, action_nums * 8).to(device)
        self.drop3 = nn.Dropout(p=0.2).to(device)

        # 根据当前的state，选取action
        self.action_head = nn.Linear(action_nums * 8, action_nums).to(device)
        # 根据当前的state，估计state value
        self.value_head = nn.Linear(action_nums * 8, 1).to(device)

    def forward(self, obs):
        res = F.leaky_relu(self.drop1(self.dense1(obs)))
        res = F.leaky_relu(self.drop2(self.dense2(res)))
        res = F.leaky_relu(self.drop3(self.dense3(res)))

        action_score = self.action_head(res)
        state_value = self.value_head(res)
        return F.softmax(action_score, dim=-1), state_value

    def select_action(self, obs):
        '''
        根据当前的状态，估计action的分布，并选取最大可能的action
        :param obs:当前的状态
        :return:action,这个action被选取时的log prob,当前状态的value估计
        '''
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action_probs, state_value = self.forward(state)
        m = Categorical(action_probs)  # action的概率分布
        action = m.sample()  # sample出一个action

        return action.item(), m.log_prob(action), state_value

    def update_policy(self):
        '''update critic'''
        policy_loss, value_loss, TD0_estimation = [], [], []
        # for each episode
        for batch_id, reward_list in enumerate(self.reward_batch):
            # for each time step
            for i, r in enumerate(reward_list):
                if i == len(reward_list) - 1:
                    # 最后一个action没有后续,TD0估计就是action的reward
                    TD0 = torch.scalar_tensor(r).to(device)
                else:
                    # TD0 = r + gamma * V[s_t+1]
                    TD0 = r + self.gamma * self.logProbs_values[batch_id][i + 1][1]
                TD0_estimation.append(TD0.detach())

        flatten_logProb_value = [prob_value for batch in self.logProbs_values for prob_value in batch]  #.to(device)

        for (logProb, value), TD0 in zip(flatten_logProb_value, TD0_estimation):
            # A[s,a]= "TD estimation of s_t" - "value of s_t"
            #       = (r + gamma * V[s_t+1]) - V[s_t]
            advantage = TD0 - value
            policy_loss.append(-1 * logProb * advantage)
            value_loss.append(F.smooth_l1_loss(TD0.reshape(-1), value.reshape(-1)))

        self.optimizer.zero_grad()
        policy_loss_sum = torch.stack(policy_loss).sum()
        policy_loss_sum.backward(retain_graph=True)
        print("\t" + str(policy_loss_sum.item()), end="")
        #保存policy network部分的变量梯度
        for name, param in self.policy_params.items():
            self.policy_grads[name] = param.grad

        self.optimizer.zero_grad()
        value_loss_sum = torch.stack(value_loss).sum()
        value_loss_sum.backward()
        print("\t" + str(value_loss_sum.item()))
        #计算梯度后，加上先前保存变量的梯度
        for name, param in self.named_parameters():
            if name in self.policy_grads:
                param.grad = param.grad + self.policy_grads[name]

        self.optimizer.step()

        del self.logProbs_values[:]
        del self.reward_batch[:]
        self.policy_grads = {}

    def save_policy(self, save_path, save_filename):
        torch.save(self.state_dict(), os.path.join(save_path, save_filename))
