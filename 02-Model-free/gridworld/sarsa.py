import numpy as np

from gridworld import GridworldEnv


class Sarsa:
    def __init__(self, env, aplha=0.01, discount=0.8,episode_nums=1000):
        self.env = env
        self.alpha = aplha
        self.time_discount = discount

        self.Q = np.zeros((env.nS, env.nA))
        self.episode_nums = episode_nums

        theta = 0.0001

    def take_action(self,env, state, action):
        res = env.P[state][action]
        return max(res, key=lambda x: x[0])

    def choose_action(self, Q_table, current_state):
        '''
        根据Q table和当前的状态选择行动
        :param Q_table: state-action value
        :param current_state: 当前所在状态
        :return: 希望的action（可以是greedy也可以是e-greedy）
        '''
        # greedy
        action = np.argmax(Q_table[current_state])
        return action

    def run(self, print_info=True):
        for ite in range(self.episode_nums):  # for each episode
            reward_total = 0
            s = np.random.randint(0,self.env.nS)
            a = self.choose_action(self.Q, s)
            while True:  # for each step in episode
                _, next_state, reward, done = self.take_action(self.env, s, a)
                a_ = self.choose_action(self.Q, next_state)
                self.Q[s][a] = self.Q[s][a] + self.alpha * (
                        reward + self.time_discount * self.Q[next_state][a_] - self.Q[s][a])
                a = a_
                s = next_state
                reward_total += reward
                if done: break

            print("in {}th step, total reward : {} ".format(ite,reward_total))
            # if np.sum(np.fabs(prev_Q - Q)) < theta: break


if __name__ == '__main__':
    env = GridworldEnv()
    sarsa = Sarsa(env,episode_nums=3000)
    sarsa.run(print_info=True)
    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(sarsa.Q, axis=1), env.shape))
    print("")
