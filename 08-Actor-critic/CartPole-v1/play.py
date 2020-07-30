import argparse

import gym
import torch
import sys
sys.path.append("..")
from models.Dense_agent import Dense_Policy

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
args = parser.parse_args()

env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

policy = Dense_Policy(input_features=4, action_nums=2,gamma=args.gamma)
policy.eval()
policy.train_step=False
policy.load_state_dict(torch.load('result.pt.tar'))

state, ep_reward, done,T = env.reset(), 0, False,0
while not done:
    action = policy.select_action(state)
    state, reward, done, _ = env.step(action)
    env.render()
    ep_reward += reward
    T+=1

print("Solved! Episode reward is now {} and the last episode runs to {} time steps!".format(ep_reward, T))
