import numpy as np

# 游戏环境
from gridworld import GridworldEnv


class policy_iteration:
    def __init__(self, env, theta=0.01, discount_factor=.8, policy=None):
        self.env = env
        self.theta = theta
        self.discount_factor = discount_factor

        if policy:
            self.policy = policy
        else:
            self.policy = self.__random_policy__()

    def __random_policy__(self):
        policy = np.zeros((self.env.nS, self.env.nA))
        for s in range(self.env.nS):
            random_action = np.random.randint(0, self.env.nA)
            policy[s, random_action] = 1.0
        return policy

    def evalue_policy(self):
        '''
        !!! Policy Evaluation !!! 从policy得到state value function
        :return: state value function
        '''
        V = np.zeros(self.env.nS)
        while True:
            # delta = 0
            pre_v = np.copy(V)
            for s in range(self.env.nS):
                res = 0
                for a in range(self.env.nA):
                    for prob, next_state, reward, done in self.env.P[s][a]:
                        res += self.policy[s][a] * prob * (reward + self.discount_factor * V[next_state])
                V[s] = res
            if np.sum(np.fabs(pre_v - V)) < self.theta:
                return V

    def __calc_q_table__(self, V):
        '''
        从state value function 中得到q table
        :param V:
        :return:
        '''
        q_table = np.zeros((self.env.nS, self.env.nA))

        for s in range(self.env.nS):
            for a in range(self.env.nA):
                res = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    res += prob * (reward + self.discount_factor * V[next_state])
                q_table[s][a] = res

        return q_table

    def update_policy(self, V):
        '''
        得到最优的policy
        :param V:
        :return:
        '''
        changed = False  # policy是否发生变化
        q_table = self.__calc_q_table__(V)

        for s in range(self.env.nS):
            best_action = np.argmax(q_table[s][:])

            # 原先的action
            old_action = np.argmax(self.policy[s][:])
            self.policy[s, old_action] = 0
            self.policy[s, best_action] = 1

            if old_action != best_action:
                changed = True

        return changed

    def get_best_policy(self):
        '''
        寻找最优的policy
        :return:
        '''
        max_iteration = 20000
        for ite in range(max_iteration):
            V = self.evalue_policy()
            changed = self.update_policy(V)
            if not changed:
                print("converged at {} steps".format(ite))
                return self.policy
        return self.policy


env = GridworldEnv()
pi = policy_iteration(env)

policy = pi.get_best_policy()

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")
