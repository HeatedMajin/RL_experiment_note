import argparse
import sys
from itertools import count

import gym
import numpy as np
import torch

sys.path.append("..")
from models.Dense_agent import Dense_Policy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='batch size (default: 32)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

policy = Dense_Policy(input_features=4, action_nums=2, gamma=args.gamma)
policy.train()
policy.train_step = True

count_ = 0
batch_reward = []
for i_episode in count(1):
    state, ep_reward = env.reset(), 0

    policy.logProbs_values.append([])
    policy.reward_batch.append([])
    for t in range(1, 10000):  # Don't infinite loop while learning
        action = policy.select_action(state)
        state, reward, done, _ = env.step(action)
        # env.render()

        #### 保存这个action的reward ####
        policy.reward_batch[-1].append(reward)
        ep_reward += reward
        if done: break

    batch_reward.append(ep_reward)

    if i_episode % args.batch_size == 0:
        top_k = 7
        batch_reward = np.array(batch_reward)
        top_k_idx = batch_reward.argsort()[::-1][0:top_k]
        avg_reward = np.mean(batch_reward[top_k_idx])
        print('Episode {}\t mean reward of the batch: {:.2f}'.format(i_episode, avg_reward))
        batch_reward = []

        if avg_reward >= env.spec.reward_threshold:
            print("Solved! Episode reward is now {} and "
                  "the last episode runs to {} time steps!".format(avg_reward, t))
            policy.save_policy(".", "result.pt.tar")
            break

        policy.update_policy()

