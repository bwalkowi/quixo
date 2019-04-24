import os
import argparse
from typing import Optional

from quixo import play
from rand_bot import Player as RandomPlayer
from utils import Mark


def train(agent_cls: str,
          src_path: Optional[str],
          dst_path: Optional[str],
          plays: int,
          gamma: float,
          epsilon: float,
          epsilon_min: float,
          epsilon_decay: float,
          learning_rate: float) -> None:

    rand_player = RandomPlayer(Mark.X)

    if agent_cls == 'dqn':
        from dqn_bot import Player as DQNPlayer
    else:
        from ddqn_bot import Player as DQNPlayer

    agent = DQNPlayer(Mark.O,
                      learning=True,
                      gamma=gamma,
                      epsilon=epsilon,
                      epsilon_min=epsilon_min,
                      epsilon_decay=epsilon_decay,
                      learning_rate=learning_rate,
                      weights_file=None)

    if src_path and os.path.isfile(src_path):
        agent.load(src_path)

    for p in range(1, plays // 2):
        _, result, rounds = play(rand_player, agent, verbose=False)
        agent.train(result)
        print(f'episode: {p}/{plays}, '
              f'score: {result.value}, '
              f'rounds: {rounds}')
    for p in range(plays // 2, plays + 1):
        result, _, rounds = play(agent, rand_player, verbose=False)
        agent.train(result)
        print(f'episode: {p}/{plays}, '
              f'score: {result.value}, '
              f'rounds: {rounds}')

    if dst_path:
        agent.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent',
                        choices=('dqn', 'ddqn'),
                        default='dqn',
                        help='Type of agent to train.')
    parser.add_argument('-l', '--load',
                        help='File containing weights.')
    parser.add_argument('-s', '--save',
                        help='File to which save learned weights.')

    parser.add_argument('-p', '--plays',
                        type=int, default=1000, metavar='P',
                        help='Number of plays')
    parser.add_argument('-g', '--gamma',
                        type=float, default=0.95, metavar='G',
                        help='Gamma hyper parameter.')
    parser.add_argument('-e', '--epsilon',
                        type=float, default=1.0, metavar='E',
                        help='Epsilon hyper parameter')
    parser.add_argument('-m', '--epsilon_min',
                        type=float, default=0.01, metavar='EM',
                        help='Epsilon_min hyper parameter')
    parser.add_argument('-d', '--epsilon_decay',
                        type=float, default=0.99, metavar='ED',
                        help='Epsilon_decay hyper parameter')
    parser.add_argument('-r', '--learning_rate',
                        type=float, default=0.001, metavar='LR',
                        help='Learning rate hyper parameter')

    args = parser.parse_args()
    train(args.agent, args.load, args.save, args.plays,
          args.gamma, args.epsilon, args.epsilon_min,
          args.epsilon_decay, args.learning_rate)


if __name__ == '__main__':
    main()
