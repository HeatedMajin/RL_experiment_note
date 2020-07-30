# -- coding: utf-8 --
import gym
import torch
from train import PPO


def main():
    #########################################################
    act_dim = 2
    obs_dim = 4
    hidden_dim = 64

    max_timestep = 2000
    lr = 1e-4
    gamma = 0.9
    eps_clip = 0.2
    seed = 1
    #########################################################

    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)

    # logging variables
    running_reward = 0

    # components of PPO algo
    ppo = PPO(obs_dim, act_dim, hidden_dim, gamma, lr, eps_clip)
    ppo.old_policy.load_state_dict(torch.load('PPO_save.pth'))

    # training step
    state = env.reset()
    for t in range(max_timestep):
        env.render()

        ## use old policy to sample data
        obs = torch.from_numpy(state).float()
        action, log_prob = ppo.old_policy.select_action(obs)
        next_state, reward, done, _ = env.step(action.item())

        state = next_state
        running_reward += reward
        if done: break

    print('finish test: reward {}'.format(running_reward))


if __name__ == '__main__':
    main()
