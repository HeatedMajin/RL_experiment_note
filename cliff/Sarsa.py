import numpy as np
from cliff_grid import GridWorld
from visualize import *

env = GridWorld()
num_states = 4 * 12  # The number of states in simply the number of "squares" in our grid world, in this case 4 * 12
num_actions = 4
alpha = 0.5
time_discount = 0.9


def choose_action(Q_table, current_state,epsilon =0.1):
    '''
    根据Q table和当前的状态选择行动
    :param Q_table: state-action value
    :param current_state: 当前所在状态
    :return: 希望的action（可以是greedy也可以是e-greedy）
    '''
    # greedy
    if np.random.random() < epsilon:
        return np.random.choice(4)
    else:
        return np.argmax(Q_table[current_state])



Q = np.zeros((num_states, num_actions))
Episode_nums = 5000

for ite in range(Episode_nums):
    s = env.reset()
    a = choose_action(Q, s)
    done = False
    while not done:
        next_state, reward, done = env.step(a)
        a_ = choose_action(Q, next_state)
        Q[s][a] = Q[s][a] + alpha * (reward + time_discount * Q[next_state][a_] - Q[s][a])
        s = next_state
        a = a_


def gen_data(visited, s):
    data = np.zeros((4, 12))
    data[-1][1:-1] = 1
    for i, j in visited:
        data[i][j] = 2
    x, y = env._id_to_position(s)
    data[x][y] = 3
    return np.flip(data,0)



done = False
s = env.reset()
visited = []
data = gen_data(visited, s)
render(data)
while not done:
    a = choose_action(Q, s)
    next_state, reward, done = env.step(a)
    s = next_state
    data = gen_data(visited, s)
    render(data)
