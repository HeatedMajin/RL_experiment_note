import numpy as np
import gym
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

def play(env,Weight):
    obs = env.reset()
    for _ in range(t_max):
        env.render()
        s = obs_to_state(env, obs)
        Q_table_ = [value_approx(s, a, Weight) for a in range(3)]
        action = np.argmax(Q_table_)
        obs, reward, done, _ = env.step(action)
        if done:
            break

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    W = np.load("mountain_car_semi_grad_weights.npy")
    play(env,W)