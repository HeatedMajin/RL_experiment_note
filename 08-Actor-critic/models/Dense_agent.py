import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical

device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")


class Dense_Policy(nn.Module):
    def __init__(self, input_features, action_nums,gamma):
        super(Dense_Policy, self).__init__()

        self.gamma = gamma
        self.eps = np.finfo(np.float64).eps.item()
        self.train_step = False

        self.build_layers(input_features, action_nums)

        ######### 每个被选取过的action保存 #############
        # 这个action的log prob和这个action的reward
        self.logProbs_values = []
        self.reward_batch = []
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-3)

    def build_layers(self, input_features, action_nums):
        self.dense1 = nn.Linear(input_features, input_features * 32).to(device)
        self.drop1 = nn.Dropout(p=0.4).to(device)
        self.dense3 = nn.Linear(input_features*32, action_nums * 8).to(device)
        self.drop3 = nn.Dropout(p=0.4).to(device)

        self.action_head = nn.Linear(action_nums * 8, action_nums).to(device)
        self.value_head = nn.Linear(action_nums * 8, 1).to(device)

    def forward(self, obs):
        res = F.leaky_relu(self.drop1(self.dense1(obs)))
        #res = F.leaky_relu(self.drop2(self.dense2(res)))
        res = F.leaky_relu(self.drop3(self.dense3(res)))
        action_score = self.action_head(res)
        state_value = self.value_head(res)
        return F.softmax(action_score, dim=-1), state_value

    def select_action(self, obs):
        state = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        action_probs, state_value = self.forward(state)
        m = Categorical(action_probs)  # action的概率分布
        action = m.sample()  # sample出一个action

        if self.train_step:
            ##### 保存选取的这个action的log prob ####
            self.logProbs_values[-1].append((m.log_prob(action), state_value))

        return action.item()

    def update_policy(self):
        policy_loss, value_loss, TD0_estimation = [], [], []
        for batch_id, batch_reward_list in enumerate(self.reward_batch):
            for i, r in enumerate(batch_reward_list):
                if i == len(batch_reward_list) - 1:
                    # 最后一个action没有后续,TD0估计就是action的reward
                    TD0 = torch.scalar_tensor(r).to(device)
                else:
                    # TD0 = r + gamma * V[s_t+1]
                    TD0 = r + self.gamma * self.logProbs_values[batch_id][i + 1][1]
                TD0_estimation.append(TD0)

        # TD0_estimation.cuda()
        flatten_log_probs = [s for batch in self.logProbs_values for s in batch]#.to(device)

        for (log_prob, value), TD0 in zip(flatten_log_probs, TD0_estimation):
            # A[s,a]= "TD estimation of s_t" - "value of s_t"
            #       = (r + gamma * V[s_t+1]) - V[s_t]
            advantage = TD0 - value
            policy_loss.append(-1 * log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(TD0.reshape(-1),value.reshape(-1)))

        self.optimizer.zero_grad()
        policy_loss_sum = torch.stack(policy_loss).sum()#.mean()
        value_loss_sum = torch.stack(value_loss).sum()#.mean()

        loss = policy_loss_sum + value_loss_sum
        print("\t"+str(loss.item()))
        loss.backward()
        self.optimizer.step()

        del self.logProbs_values[:]
        del self.reward_batch[:]

    def save_policy(self, save_path, save_filename):
        torch.save(self.state_dict(), os.path.join(save_path, save_filename))
