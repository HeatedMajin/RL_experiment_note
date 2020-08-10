import gym
import numpy as np

n_states = 40
iter_max = 5000

gamma = 0.98
t_max = 1000
eps = 0.1
lr = 0.01


def obs_to_state(env, obs):
    """ Maps an observation to state """
    segmentation_factor = 100
    pos_segment = (env.high[0] - env.low[0]) / segmentation_factor
    vel_segment = (env.high[1] - env.low[1]) / segmentation_factor

    coarse_state = np.zeros(2 * segmentation_factor)

    coarse_state[int((obs[0] - env.low[0]) / pos_segment)] = 1

    coarse_state[int((obs[1] - env.low[1]) / vel_segment) + segmentation_factor] = 1

    return coarse_state


def value_approx(state, action, weights):
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    s_a = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            s_a[w_i] = s_i * a_i
            w_i = w_i + 1

    return np.dot(s_a, weights)


def value_approx_grad(env, state, action, weights):
    """
    It calculates the gradient of the state-action pair.
    """
    action_one_hot_vector = np.zeros(env.action_space.n)
    action_one_hot_vector[action] = 1
    gradient = np.zeros(len(weights))

    w_i = 0
    for s_i in state:
        for a_i in action_one_hot_vector:
            gradient[w_i] = s_i * a_i
            w_i = w_i + 1

    return gradient


def choose_action(env, Q_table, epsilon=0.2):
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
        return np.argmax(Q_table)


def play(env):
    obs = env.reset()
    for _ in range(t_max):
        env.render()
        s = obs_to_state(env, obs)
        Q_table_ = [value_approx(s, a, W) for a in range(3)]
        action = np.argmax(Q_table_)
        obs, reward, done, _ = env.step(action)
        if done:
            break


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    s = env.reset()

    num_episodes = 5000
    done = False
    env_dim = len(obs_to_state(env, s))
    W = np.zeros(env_dim * env.action_space.n)

    for i_episode in range(1, num_episodes + 1):
        total_reward = 0

        # init state
        s = obs_to_state(env, env.reset())
        Q_table = [value_approx(s, a, W) for a in range(3)]
        # init action
        a = choose_action(env, Q_table)

        Q_pre = Q_table[a]  # q(S,A,W)

        for t in range(t_max):
            # take action A, observe R and S'
            next_obs, r, done, _ = env.step(a)
            next_s = obs_to_state(env, next_obs)

            total_reward += r

            # s' is terminal
            if done:
                W = W + lr * (r - Q_pre) * value_approx_grad(env, s, a, W)
                break

            # estimate q(s',w) , choose A'
            Q_table_next = [value_approx(next_s, a, W) for a in range(3)]
            next_a = choose_action(env, Q_table_next)

            # update weights
            Q_next_a = Q_table_next[next_a]  # q(S',A',W)
            W = W + lr * (r + gamma * Q_next_a - Q_pre) * value_approx_grad(env, s, a, W)

            s, a = next_s, next_a
            Q_pre = Q_next_a

        if i_episode % 10 == 0:
            print('Iteration #%d -- Total reward = %d.' % (i_episode, total_reward))
    np.save("mountain_car_semi_grad_weights.npy", W)
    play(env)
