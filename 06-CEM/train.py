import argparse

import gym
import numpy as np
from gym import logger
import pickle

class BinaryActionLinearPolicy(object):
    def __init__(self, theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def act(self, ob):
        y = ob.dot(self.w) + self.b
        a = int(y < 0)
        return a


def cem(f, th_mean, th_std, batch_size, elite_frac):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean:mean over input distribution
    th_std:std over input distribution
    batch_size: number of samples of theta to evaluate per batch
    elite_frac: each batch, select this fraction of the top-performing samples
    """
    n_elite = int(np.round(batch_size * elite_frac))

    ths = np.array(
        [th_mean + dth for dth in np.expand_dims(th_std, axis=0) * np.random.randn(batch_size, th_mean.size)])
    ys = np.array([f(th) for th in ths])  # 以每个theta的policy，会得到多少的收益
    elite_inds = ys.argsort()[::-1][:n_elite]
    elite_ths = ths[elite_inds]
    th_mean = elite_ths.mean(axis=0)
    th_std = elite_ths.std(axis=0)
    return th_mean, th_std, ys.mean()


def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t % 3 == 0: env.render()
        if done:
            break
    return total_rew, t + 1


if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default="CartPole-v1")
    args = parser.parse_args()

    env = gym.make(args.target)
    env.seed(0)
    np.random.seed(0)
    params = dict(batch_size=100, elite_frac=0.2)
    num_steps = 500

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    #初始化：theta所在分布的均值和方差 N(0,1)
    th_mean = np.zeros(env.observation_space.shape[0] + 1) # w + 1
    th_std = np.ones_like(th_mean) * 1.

    # Train the agent, and snapshot each stage
    for i in range(200):
        th_mean, th_std, ys_mean = cem(noisy_evaluation, th_mean, th_std, **params)

        print('Iteration %2i. Episode mean reward: %7.3f' % (i, ys_mean))

    env.close()
    with open("save/theta_mean_std.tar","wb") as f:
        pickle.dump([th_mean,th_std],f)


