import numpy as np
import torch
from torch import optim

from Bird_DQN import *


class Agent():
    def __init__(self, config):
        super(Agent, self).__init__()
        self.device = config.device

        self.gamma = config.GAMMA
        self.lr = config.LR

        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        #self.learn_start = config.LEARN_START

        self.actions_num = 2

        self.declear_network()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        #Adam(self.model.parameters(), lr=self.lr)

        self.model.to(self.device)
        self.target_model.to(self.device)

        ##########启用 batch normalization 和 dropout???????????
        self.model.train()
        self.target_model.train()

        self.update_count = 0

        self.declear_memory(config.EXP_REPLAY_SIZE)

    def declear_network(self):
        self.model = Bird_DQN(self.actions_num)
        self.target_model = Bird_DQN(self.actions_num)

    def declear_memory(self, mem_size):
        self.mem = ReplayMemory(mem_size)

    def store_memory(self, s, a, r, s_):
        self.mem.push((s, a, r, s_))

    def update_Q(self, obs, action, reward, next_obs):
        self.store_memory(obs, action, reward, next_obs)

        state_batch, action_batch, reward_batch, next_state_batch = self.prepare_minibatch()
        loss = self.compute_loss(state_batch, action_batch, reward_batch, next_state_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_count += 1

        # w_ update after 50 times of w's update
        if self.update_count % self.target_net_update_freq == 0:
            self.update_count = 0
            self.update_target_model()

    def compute_loss(self, state_batch, action_batch, reward_batch, next_state_batch):

        # estimate
        #current_q_values = self.model(state_batch).gather(1, torch.argmax(action_batch, dim=1).unsqueeze(dim=1))
        # target
        # with torch.no_grad():
        #     max_next_action = self.get_max_next_state_action(next_state_batch)
        #     max_next_q_values = self.target_model(next_state_batch).gather(1, max_next_action)
        #     expected_q_values = reward_batch.view(-1, 1) + (self.gamma * max_next_q_values)

        # estimate
        current_q_values = self.model(state_batch).gather(1, torch.argmax(action_batch, dim=1).unsqueeze(dim=1))
        # target
        with torch.no_grad():
            max_next_q_values = self.target_model(next_state_batch).max(1)[0]

        expected_q_values = reward_batch.unsqueeze(dim=1) + (self.gamma * max_next_q_values.unsqueeze(dim=1))

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def prepare_minibatch(self):
        minibatch = self.mem.sample(self.batch_size)
        state_batch = torch.tensor(np.array([data[0] for data in minibatch]), device=self.device)#.unsqueeze(dim=1)
        action_batch = torch.tensor(np.array([data[1] for data in minibatch]), device=self.device)
        reward_batch = torch.tensor(np.array([data[2] for data in minibatch]), device=self.device)
        next_state_batch = torch.tensor(np.array([data[3] for data in minibatch]), device=self.device)#.unsqueeze(dim=1)

        return state_batch, action_batch, reward_batch, next_state_batch

    def get_action(self, obs, epsilon):
        #return self.optimal_action(obs)
        if np.random.random() < epsilon:
            return self.random_action()
        else:
            return self.optimal_action(obs)

    def optimal_action(self, obs):
        with torch.no_grad():
            X = torch.tensor([obs], device=self.device, dtype=torch.float)
            a = self.model(X).argmax(1)

            action = np.zeros(self.actions_num, dtype=np.float32)
            action[a.item()] = 1
            return action

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).argmax(dim=1).view(-1, 1)

    def random_action(self):
        action = np.zeros(self.actions_num, dtype=np.float32)
        action_index = 0 if np.random.random() < 0.8 else 1
        action[action_index] = 1
        return action

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
