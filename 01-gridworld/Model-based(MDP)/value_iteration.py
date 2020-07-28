import numpy as np

# 游戏环境
from gridworld import GridworldEnv


class value_iteration:
    def __init__(self, env, theta=0.001, discount_factor=1.0):
        self.env = env
        self.theta = theta
        self.discount_factor = discount_factor

    def __calc_Q_table__(self, current_s, V):
        # 四种行为
        Q_curS = np.zeros(env.nA)
        for action in range(env.nA):
            for prob, next_state, reward, done in env.P[current_s][action]:
                Q_curS[action] += prob * (reward + self.discount_factor * V[next_state])
        return Q_curS

    def calc_V_table(self):
        # 16个状态的value
        V = np.zeros(env.nS)
        while True:
            delta = -1e10  # 记录V表的最大变化，小于theta时，说明所有的更新都很小，结束
            for current_s in range(env.nS):
                # 在当前的state下，计算Q table，Q[s][a]
                Q_curS = self.__calc_Q_table__(current_s, V)

                # Q table中，最大的Q[s][a] 赋值给V[s]   (bellman optimal equation!!!!!!!)
                best_action_value = np.max(Q_curS)

                delta = max(delta, np.abs(best_action_value - V[current_s]))  # 判断是否还要继续

                V[current_s] = best_action_value
            if (delta < self.theta):
                return V

    def get_policy(self, V):
        # policy[s][a]表示s状态下，a行为采取的概率
        policy = np.zeros([env.nS, env.nA])
        for current_s in range(env.nS):
            # 在当前的状态下，计算Q table，Q[s][a]
            Q_curS = self.__calc_Q_table__(current_s, V)

            # 找到最优的action, v(s) = max_a q(s,a)
            best_action = np.argmax(Q_curS)

            # 总是采取最优的action
            policy[current_s, best_action] = 1.0
        return policy


env = GridworldEnv()
vi = value_iteration(env)

value_table = vi.calc_V_table()
policy = vi.get_policy(value_table)

print(value_table)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")
