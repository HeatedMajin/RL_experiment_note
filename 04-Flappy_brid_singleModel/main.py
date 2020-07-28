import argparse

import MSIC
from Brid_DQN import *

parser = argparse.ArgumentParser(description='DQN demo for flappy bird')
parser.add_argument('--train', action='store_true', default=False, help='Train the model or play with pretrained model')
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--gamma', type=float, help='discount rate', default=0.99)
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--memory_size', type=int, help='memory size for experience replay', default=5000)
parser.add_argument('--init_e', type=float, help='initial epsilon for epsilon-greedy exploration', default=1)
parser.add_argument('--final_e', type=float, help='final epsilon for epsilon-greedy exploration', default=0.1)
parser.add_argument('--observation', type=int, help='random observation number before training', default=100)
parser.add_argument('--exploration', type=int, help='number of exploration using epsilon-greedy policy', default=10000)
parser.add_argument('--max_episode', type=int, help='maximum episode of training', default=20000)
parser.add_argument('--weight', type=str, help='weight file name for finetunig(Optional)', default='')
parser.add_argument('--save_checkpoint_freq', type=int, help='episode interval to save checkpoint', default=1000)

if __name__ == '__main__':
    args = parser.parse_args()
    if  args.train:
        model = Bird_DQN(epsilon=args.init_e, mem_size=args.memory_size)
        args.resume = not args.weight == ""  # 是否在预先训练的模型上训练
        MSIC.train(model, args)
    else:
        MSIC.play("checkpoint-episode-7000.pth.tar")
