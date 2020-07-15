import sys

sys.path.append("game/")
import wrapped_flappy_bird as game
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import PIL.Image as Image
from torch.autograd import Variable
from Brid_DQN import *
import shutil

IMAGE_SIZE = (72, 128)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename, model):
    try:
        checkpoint = torch.load(filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    if 'episode' in checkpoint:
        episode = checkpoint['episode']
    else:
        episode = checkpoint['episode:']
    epsilon = checkpoint['epsilon']
    print('pretrained episode = {}'.format(episode))
    print('pretrained epsilon = {}'.format(epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    time_step = checkpoint.get('best_timestep', None)
    if time_step is None:
        time_step = checkpoint['time_step']
    print('pretrained time step = {}'.format(time_step))
    return episode, epsilon, time_step


def preprocess(frame):
    """Do preprocessing: resize and binarize.

       Downsampling to 128x72 size and convert to grayscale
       frame -- input frame, rgb image with 512x288 size
    """
    im = Image.fromarray(frame).resize(IMAGE_SIZE).convert(mode='L')
    out = np.asarray(im).astype(np.float32)
    out[out <= 1.] = 0.0
    out[out > 1.] = 1.0
    return out


def train(model, options, resume):
    if resume:
        if options.weight is None:
            print('when resume, you should give weight file name.')
            return
        print('load previous model weight: {}'.format(options.weight))
        _, _, best_time_step = load_checkpoint(options.weight, model)

    flappyBird = game.GameState()

    best_timestep = 0
    optimizer = optim.RMSprop(model.parameters(), lr=options.lr)
    ceriterion = nn.MSELoss()

    action = [1, 0]  # do nothing
    obs, reward, terminal = flappyBird.frame_step(action)
    obs = preprocess(obs)
    model.set_initial_state()

    # in the first `OBSERVE` time steos, we dont train the model
    for i in range(options.observation):
        action = model.random_action()
        obs_next, r, terminal = flappyBird.frame_step(action)
        obs_next = preprocess(obs_next)
        model.store_trans(action=action, reward=r, next_obs=obs_next, finish=terminal)

    # start training
    for episode in range(options.max_episode):
        model.time_step = 0
        model.set_trainable(True)
        total_reward = 0.

        # begin an episode!
        while True:
            optimizer.zero_grad()

            # e-greedy to choose an action
            action = model.take_action(obs)
            o_next, r, terminal = flappyBird.frame_step(action)
            total_reward += options.gamma ** model.time_step * r
            o_next = preprocess(o_next)
            model.store_trans(action, r, o_next, terminal)
            model.increase_timestep()

            # Step 1: obtain random minibatch from replay memory
            minibatch = random.sample(model.replay_mem, options.batch_size)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])

            state_batch_var = Variable(torch.from_numpy(state_batch))
            next_state_batch_var = Variable(torch.from_numpy(next_state_batch), requires_grad=False)

            # Step 2: calculate y
            q_value_next = model.forward(next_state_batch_var)  # S'下的所有action的Q table项
            q_value = model.forward(state_batch_var)  # S 下的所有action的Q table项
            max_q, _ = torch.max(q_value_next, dim=1)  # S'下的所有action的Q table项中的最大值

            # Bellman optimal equation : V[s] = max_a( reward[s,a] + gamma * Q[s,a] )
            y = reward_batch.astype(np.float32)
            for i in range(options.batch_size):
                if not minibatch[i][4]:
                    y[i] += options.gamma * max_q.data[i]
            y = Variable(torch.from_numpy(y))

            # Q table和V func的关系：V[s] = sum_a ( p(a|s) * q[s,a] )
            action_batch_var = Variable(torch.from_numpy(action_batch))
            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

            loss = ceriterion(q_value, y)
            loss.backward()
            optimizer.step()

            # when the bird dies, the episode ends
            if terminal:
                break

        # 当前一轮，轮数、探索率、最大存活时间、总的收益
        print('episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
            episode, model.epsilon, model.time_step, total_reward))

        # 调整epsilon：先前很大、往后变小
        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e) / options.exploration
            model.epsilon -= delta

        # 每过100轮，测试一次模型
        if episode % 100 == 0:
            ave_time = test_dqn(model, episode)

        # 根据测试结果，保存最优的模型
        if ave_time > best_timestep:
            best_timestep = ave_time
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'best_timestep': best_timestep,
            }, True, 'checkpoint-episode-%d.pth.tar' % episode)
        elif episode % options.save_checkpoint_freq == 0:
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'time_step': ave_time,
            }, False, 'checkpoint-episode-%d.pth.tar' % episode)
        else:
            continue

        print('save checkpoint, episode={}, ave time step={:.2f}'.format(
            episode, ave_time))


def test_dqn(model, episode):
    """Test the behavor of dqn when training

       model -- dqn model
       episode -- current training episode
    """
    model.set_trainable(False)
    ave_time = 0.
    for test_case in range(5):
        model.time_step = 0
        flappyBird = game.GameState()
        o, r, terminal = flappyBird.frame_step([1, 0])
        o = preprocess(o)
        while True:
            action = model.optimal_action()
            o, r, terminal = flappyBird.frame_step(action)
            if terminal:
                break
            o = preprocess(o)
            model.current_state = np.append(model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)
            model.increase_timestep()
        ave_time += model.time_step
    ave_time /= 5
    print('testing: episode: {}, average time: {}'.format(episode, ave_time))
    return ave_time


def play(model_file_name):
    print('load pretrained model file: ' + model_file_name)
    model = Bird_DQN(epsilon=0., mem_size=0)
    load_checkpoint(model_file_name, model)

    model.set_trainable(False)
    bird_game = game.GameState()
    model.set_initial_state()
    while True:
        action = model.optimal_action()
        o, r, terminal = bird_game.frame_step(action)
        if terminal:
            break
        o = preprocess(o)

        model.current_state = np.append(model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)

        model.increase_timestep()

    print('total time step is {}'.format(model.time_step))