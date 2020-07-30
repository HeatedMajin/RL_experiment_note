# -- coding: utf-8 --
import gym
import torch
from torch import nn
from torch.distributions import Categorical


class Exprience_Memory():
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logProbs = []

    def save(self, state, action, reward, done, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.logProbs.append(log_prob)

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logProbs[:]

    def __len__(self):
        assert len(self.states) == len(self.actions)
        assert len(self.actions) == len(self.rewards)
        assert len(self.rewards) == len(self.is_terminals)
        assert len(self.is_terminals) == len(self.logProbs)
        return len(self.states)


class Actor_critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super(Actor_critic, self).__init__()

        self.action_nn = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )
        self.value_nn = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self):
        raise NotImplementedError

    def select_action(self, obs):
        actions_prob = self.action_nn(obs)

        m = Categorical(actions_prob)  # action的概率分布
        action = m.sample()  # sample出一个action

        return action, m.log_prob(action)

    def evaluate(self, obs, action):
        value = self.value_nn(obs)
        actions_prob = self.action_nn(obs)
        m = Categorical(actions_prob)  # action的概率分布

        return value.squeeze(), m.log_prob(action)


class PPO():
    def __init__(self, obs_dim, act_dim, hidden_dim, gamma, lr, eps_clip):
        self.old_policy = Actor_critic(obs_dim, act_dim, hidden_dim)
        self.cur_policy = Actor_critic(obs_dim, act_dim, hidden_dim)

        self.optim = torch.optim.Adam(self.cur_policy.parameters(), lr=lr)
        self.old_policy.load_state_dict(self.cur_policy.state_dict())
        self.mseloss = nn.MSELoss()

        self.gamma = gamma
        self.eps_clip = eps_clip

    def update(self, mem):
        ## MC estimation of reward
        returns = []
        discount_reward = 0
        for r, done in zip(reversed(mem.rewards), reversed(mem.is_terminals)):
            if done: discount_reward = 0
            discount_reward = r + self.gamma * discount_reward
            returns.insert(0, discount_reward)

        ## nomalization return
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        old_logProbs = torch.stack(mem.logProbs).detach()
        old_states = torch.stack(mem.states).detach()
        old_actions = torch.stack(mem.actions).detach()

        for _ in range(4):
            value, log_prob_action = self.cur_policy.evaluate(old_states, old_actions)

            advantage = returns - value

            ratios = torch.exp(log_prob_action - old_logProbs.detach())

            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage

            object_function = torch.min(surr1, surr2).mean() - 0.5 * self.mseloss(value, returns)
            loss = - object_function

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.old_policy.load_state_dict(self.cur_policy.state_dict())


def main():
    #########################################################
    act_dim = 2
    obs_dim = 4
    hidden_dim = 64

    max_episode = 100000
    max_timestep = 2000
    update_timestep = 200  # update when the memory is this size

    log_interval =10
    lr = 1e-4
    gamma = 0.9
    eps_clip = 0.2
    seed = 1
    solved_reward = 200
    #########################################################

    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)

    # logging variables
    running_reward = 0
    avg_length = 0

    # components of PPO algo
    ppo = PPO(obs_dim, act_dim, hidden_dim, gamma, lr, eps_clip)
    mem = Exprience_Memory()

    # training step
    for i_episode in range(max_episode):
        state = env.reset()
        for t in range(max_timestep):
            ## use old policy to sample data
            obs = torch.from_numpy(state).float()
            action, log_prob = ppo.old_policy.select_action(obs)
            next_state, reward, done, _ = env.step(action.item())
            mem.save(obs, action, reward, done, log_prob)

            ## update cur policy and old policy
            if (len(mem) == update_timestep):
                ppo.update(mem)
                mem.clear()

            state = next_state
            running_reward += reward
            if done: break

        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward >= (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.cur_policy.state_dict(), './PPO_save.pth')
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))

            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
if __name__ == '__main__':
    main()
