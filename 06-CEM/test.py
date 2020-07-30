import argparse
import pickle

import gym
import numpy as np
from gym import logger
from train import BinaryActionLinearPolicy


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    env = gym.make("CartPole-v1")
    env.seed(0)
    np.random.seed(0)

    with open("save/theta_mean_std.tar", "rb") as f:
        theta_mean, theta_std = pickle.load(f)

    agent = BinaryActionLinearPolicy(theta_mean)
    total_rew = 0
    ob = env.reset()

    t, done = 0, False
    while not done:
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        env.render()
        t += 1

    print('survival times %2i. Episode mean reward: %7.3f' % (t, total_rew))
