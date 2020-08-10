import argparse
import sys
from itertools import count

import gym
import numpy as np
import torch

sys.path.append("..")
from models.Dense_agent2 import Dense_Policy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.95, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--batch_size', type=int, default=10, metavar='N', help='batch size (default: 32)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

# 创建游戏环境
env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

# 定义policy
policy = Dense_Policy(input_features=4, action_nums=2, gamma=args.gamma)

#policy.load_state_dict(torch.load('result.pt.tar'))
policy.train()
policy.train_step = True

# train step
batch_reward = []
# for each episode
for i_episode in count(1):
    state, ep_total_reward = env.reset(), 0

    policy.logProbs_values.append([])
    policy.reward_batch.append([])

    # step 1：收集要训练的数据
    for t in range(1, 10000):  # Don't infinite loop while learning
        action, logProb, stateValue = policy.select_action(state)
        next_state, reward, done, _ = env.step(action)

        #### 保存这个action的reward,logProb 和当前的state value ####
        policy.reward_batch[-1].append(reward)
        policy.logProbs_values[-1].append((logProb, stateValue))

        state = next_state
        ep_total_reward += reward
        if done: break
    batch_reward.append(ep_total_reward)

    # step 2：收集满一个bacth的数据后，训练
    if i_episode % args.batch_size == 0:
        # 看最好的K个数据是否满足停止条件
        top_k = 5
        batch_reward = np.array(batch_reward)
        top_k_idx = batch_reward.argsort()[::-1][0:top_k]
        avg_reward = np.mean(batch_reward[top_k_idx])
        print('Episode {}\t TOP@{}\'s mean reward of the batch: {:.2f}'.format(i_episode, top_k, avg_reward),end="")
        batch_reward = []

        if avg_reward >= env.spec.reward_threshold:
            print("Solved! Episode reward is {}, the last episode runs to {} time steps!".format(avg_reward, t))
            policy.save_policy(".", "result.pt.tar")
            break

        # 为满足停止条件，更新policy
        policy.update_policy()
