import argparse
import shutil
import sys

import PIL.Image as Image

from Agent import *
from utils.hyperparameters import Config

sys.path.append("game/")
import wrapped_flappy_bird as game

parser = argparse.ArgumentParser(description='DQN demo for flappy bird')
parser.add_argument('--train', action='store_true', default=False, help='Train the model or play with pretrained model')

# 处理图片
IMAGE_SIZE = (128, 72)


# 512 288
def preprocess(frame):
    """Do preprocessing: resize and binarize.

       Downsampling to 128x72 size and convert to grayscale
       frame -- input frame, rgb image with 512x288 size
    """
    im = Image.fromarray(frame).crop((0, 0, 400, 288)).resize(IMAGE_SIZE).convert(mode='L')
    out = np.asarray(im).astype(np.float32)
    out[out <= 1.] = 0.0
    out[out > 1.] = 1.0
    return out


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, "check_point/" + filename)
    if is_best:
        shutil.copyfile("check_point/" + filename, 'check_point/model_best.pth.tar')


def load_checkpoint(filename, model):
    try:
        checkpoint = torch.load("check_point/" + filename)
    except:
        # load weight saved on gpy device to cpu device
        # see https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print('pretrained episode = {}'.format(episode))
    print('pretrained epsilon = {}'.format(epsilon))
    model.load_state_dict(checkpoint['state_dict'])
    time_step = checkpoint.get('best_timestep', None)
    if time_step is None:
        time_step = checkpoint['time_step']
    print('pretrained time step = {}'.format(time_step))
    return episode, epsilon, time_step


def test_dqn(model, episode):
    ave_time = 0.
    for test_case in range(5):
        time_step = 0
        flappyBird = game.GameState()
        o, r, terminal = flappyBird.frame_step([1, 0])
        obs = preprocess(o)
        state = init_state()
        state = np.append(state[1:, :, :], obs.reshape((1,) + obs.shape), axis=0)

        while True:
            action = model.optimal_action(state)
            o, r, terminal = flappyBird.frame_step(action)
            if terminal: break
            o = preprocess(o)
            state = np.append(state[1:, :, :], o.reshape((1,) + o.shape), axis=0)
            time_step += 1
        ave_time += time_step
    ave_time /= 5
    print('testing: episode: {}, average time: {}'.format(episode, ave_time))
    return ave_time


def play(model_file_name, config):
    print('load pretrained model file: ' + model_file_name)

    agent = Agent(config)
    load_checkpoint(model_file_name, agent.model)
    bird_game = game.GameState()

    total_reward = 0.
    time_count = 0.

    # 1.init S
    action = [1, 0]  # do nothing
    state = init_state()
    obs, reward, terminal = bird_game.frame_step(action)
    obs = preprocess(obs)
    state = np.append(state[1:, :, :], obs.reshape((1,) + obs.shape), axis=0)

    while not terminal:
        action = agent.optimal_action(state)

        next_obs, reward, terminal = bird_game.frame_step(action)
        next_obs = preprocess(next_obs)
        next_state = np.append(state[1:, :, :], next_obs.reshape((1,) + next_obs.shape), axis=0)

        state = next_state

        total_reward += reward
        time_count += 1

    print('total time step is {}'.format(time_count))


def init_state():
    empty_frame = np.zeros((72, 128), dtype=np.float32)
    empty_state = np.stack((empty_frame, empty_frame, empty_frame), axis=0)

    return empty_state


def train(agent, config):
    if config.resume:
        episode_start, epsilon, best_timestep = load_checkpoint(config.resume_file, agent.model)
    else:
        episode_start, best_timestep = 0, 0

    # 1.init S
    action = [1, 0]  # do nothing
    state = init_state()
    obs, reward, terminal = env.frame_step(action)
    obs = preprocess(obs)
    state = np.append(state[1:, :, :], obs.reshape((1,) + obs.shape), axis=0)

    # in the first `OBSERVE` time steps, we dont train the model
    for i in range(config.LEARN_START_FRAME):
        action = agent.random_action()
        obs_next, r, terminal = env.frame_step(action)
        obs_next = preprocess(obs_next)
        state_next = np.append(state[1:, :, :], obs_next.reshape((1,) + obs_next.shape), axis=0)

        agent.store_memory(s=state, a=action, r=r, s_=state_next)
        if (terminal):
            state = init_state()
        else:
            state = state_next

    # train the agent
    for episode_idx in range(1, config.MAX_EPISODE + 1):
        total_reward = 0.
        time_count = 0
        epsilon = config.epsilon_by_frame(frame_idx=episode_idx)

        while True:
            # 2.choose A from s using policy derived from Q(e-greedy)
            action = agent.get_action(state, epsilon)

            # 3.take action A, observe R and s'
            next_obs, reward, terminal = env.frame_step(action)
            next_obs = preprocess(next_obs)
            next_state = np.append(state[1:, :, :], next_obs.reshape((1,) + next_obs.shape), axis=0)

            # 4.update Q
            agent.update_Q(state, action, reward, next_state)

            # 5. s <-- s'
            state = next_state

            total_reward += reward
            time_count += 1

            if terminal: break

        # 当前一轮，轮数、探索率、最大存活时间、总的收益
        print('episode: {}, epsilon: {:.4f}, max timestep: {}, total reward: {:.6f}'.format(
            episode_idx, epsilon, time_count, total_reward))

        ave_time = 0
        # 每过200轮，测试一次模型
        if episode_idx % 200 == 0:
            ave_time = test_dqn(agent, episode_idx)

        # 根据测试结果，保存最优的模型
        if ave_time > best_timestep:
            best_timestep = ave_time
            save_checkpoint({
                'episode': episode_idx,
                'epsilon': epsilon,
                'state_dict': agent.model.state_dict(),
                'best_timestep': best_timestep,
            }, True, 'checkpoint-episode-%d.pth.tar' % episode_idx)
        elif episode_idx % config.save_checkpoint_freq == 0:
            save_checkpoint({
                'episode': episode_idx,
                'epsilon': epsilon,
                'state_dict': agent.model.state_dict(),
                'time_step': ave_time,
            }, False, 'checkpoint-episode-%d.pth.tar' % episode_idx)
        else:
            continue

        print('save checkpoint, episode={}, ave time step={:.2f}'.format(episode_idx, ave_time))


if __name__ == '__main__':
    env = game.GameState()
    args = parser.parse_args()
    config = Config()
    config.resume_file = ""
    config.resume = (config.resume_file != "")

    if True:#args.train:
        agent = Agent(config)
        train(agent, config)
    else:
        play("checkpoint-episode-28000.pth.tar", config)
