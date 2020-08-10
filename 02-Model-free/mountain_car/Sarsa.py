import gym
import numpy as np

n_states = 40
iter_max = 5000

initial_lr = 1.0  # Learning rate
min_lr = 0.003
gamma = 0.9
t_max = 10000
eps = 0.1
lr = 0.1


def obs_to_state(env, obs):
    """ Maps an observation to state """
    # we quantify the continous state space into discrete space
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0]) / env_dx[0])
    b = int((obs[1] - env_low[1]) / env_dx[1])
    return a, b


def choose_action(env, Q_table, current_state, epsilon=0.1):
    '''
    根据Q table和当前的状态选择行动
    :param Q_table: state-action value
    :param current_state: 当前所在状态
    :return: 希望的action（可以是greedy也可以是e-greedy）
    '''
    # greedy
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[current_state[0]][current_state[1]])
        # return np.argmax(Q_table[current_state])


def play(env):
    obs = env.reset()
    for _ in range(t_max):
        env.render()
        a, b = obs_to_state(env, obs)
        action = np.argmax(q_table[a][b])
        obs, reward, done, _ = env.step(action)
        if done:
            break


q_table = np.zeros((n_states, n_states, 3))

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    num_episodes = 2000
    done = False
    for i_episode in range(num_episodes):
        eta = max(min_lr, initial_lr * (0.85 ** (i_episode // 100)))

        total_reward = 0
        observation = env.reset()
        a, b = obs_to_state(env, observation)
        action = choose_action(env, q_table, current_state=(a, b))

        for t in range(t_max):
            next_obs, reward, done, _ = env.step(action)

            a_, b_ = obs_to_state(env, next_obs)
            action_next = choose_action(env, q_table, current_state=(a_, b_))


            q_table[a][b][action] = q_table[a][b][action] + eta * (
                    reward + gamma * q_table[a_][b_][action_next] - q_table[a][b][action])

            a, b, action = a_, b_, action_next
            total_reward += reward

            if done:
                break
        if i_episode % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i_episode + 1, total_reward))
    play(env)
