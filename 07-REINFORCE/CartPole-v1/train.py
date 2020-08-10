import argparse
import sys
from itertools import count

import gym
import torch

sys.path.append("..")
from models.Dense_agent import Dense_Policy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

policy = Dense_Policy(input_features=4, action_nums=2)
policy.train_step = True

count_ = 0

for i_episode in count(1):
    state, ep_reward = env.reset(), 0
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = policy.select_action(state)
        state, reward, done, _ = env.step(action)

        #### 保存这个action的reward ####
        policy.reward_pre_action.append(reward)
        ep_reward += reward
        if done: break

    policy.update_policy()

    print('Episode {}\tLast reward: {:.2f}'.format(i_episode, ep_reward))

    if ep_reward > env.spec.reward_threshold:
        count_ += 1
        if count_ == 100:  # 为了减少波动产生的影响
            print("Solved! Episode reward is now {} and "
                  "the last episode runs to {} time steps!".format(ep_reward, t))

            policy.save_policy(".", "result.pt.tar")
            break
    pre_reward = ep_reward
