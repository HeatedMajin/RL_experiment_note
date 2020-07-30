import argparse
import sys

import PIL.Image as Image
import gym
import numpy as np
import torch

sys.path.append("..")
from models.DCN_agent import DCN_policy

from itertools import count

parser = argparse.ArgumentParser(description='PyTorch policy gradient example at openai-gym pong')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99')
parser.add_argument('--decay_rate', type=float, default=0.99, metavar='G',
                    help='decay rate for RMSprop (default: 0.99)')
parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='G',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--batch_size', type=int, default=5, metavar='G',
                    help='Every how many episodes to da a param update')
parser.add_argument('--seed', type=int, default=87, metavar='N',
                    help='random seed (default: 87)')
parser.add_argument('--test', action='store_true',
                    help='whether to test the trained model or keep training')

args = parser.parse_args()

"""environment"""
env = gym.make('Pong-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

def preprocess(obs):
    '''
    :param obs: 210,160,3
    :return:60,60
    '''
    Image.fromarray(obs)
    IMAGE_SIZE = (60, 60)
    obs = obs[:, :, 0]
    im = Image.fromarray(obs).crop((0, 30, 160, 200)).resize(IMAGE_SIZE).convert(mode='L')
    out = np.asarray(im).astype(np.float32)
    out[out == 142] = 0
    out[out == 144] = 0
    out[out == 143] = 0
    out[out == 145] = 0

    out[out == 109] = 0
    out[out == 110] = 0
    out[out == 108] = 0
    out[out != 0] = 1
    return out

policy = DCN_policy(2)
policy.load_state_dict(torch.load('pg_params6700.pkl'))
policy.train_step = False
policy.eval()

reward_sum = 0
obs = env.reset()
state = preprocess(obs)
for t in range(20000):
    env.render()
    action = policy.take_action(state)
    env_action = action + 2
    obs, reward, done, _ = env.step(env_action)
    state = preprocess(obs)

    policy.rewards.append(reward)

    reward_sum += reward
    if done:
        print('reward: %f. ' % (reward_sum))
        break