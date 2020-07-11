import numpy as np

from gridworld import GridworldEnv


def take_action(env, state, action):
    res = env.P[state][action]
    return max(res, key=lambda x: x[0])


def choose_action(Q_table, current_state):
    '''
    根据Q table和当前的状态选择行动
    :param Q_table: state-action value
    :param current_state: 当前所在状态
    :return: 希望的action（可以是greedy也可以是e-greedy）
    '''
    # greedy
    action = np.argmax(Q_table[current_state])
    return action


alpha = 0.01
time_discount = 0.8
episode_nums = 7000
env = GridworldEnv()

Q = np.zeros((env.nS, env.nA))

for i in range(episode_nums):  # for each episode
    s = np.random.randint(0, env.nS)

    done = False
    while not done:
        a = choose_action(Q, s)
        _, next_state, reward, done = take_action(env, s, a)
        Q[s][a] = Q[s][a] + alpha * (reward + time_discount * max(Q[next_state]) - Q[s][a])
        s = next_state
print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(Q, axis=1), env.shape))
print("")