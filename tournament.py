import argparse

from quixo import play
from utils import Mark
from rand_bot import Player as RandomPlayer


def play_tournament(plays: int, agent: str, weights_file: str) -> None:
    if agent == 'dqn':
        from dqn_bot import Player as NNPlayer
    elif agent == 'ddqn':
        from ddqn_bot import Player as NNPlayer
    else:
        from dddqn_bot import Player as NNPlayer

    p1 = RandomPlayer(Mark.X)
    p2 = NNPlayer(Mark.O, weights_file=weights_file, epsilon=0.05)

    for p in range(1, plays+1, 2):
        _, result, rounds = play(p1, p2, verbose=False)
        print(f'episode: {p:>2}/{plays}, result: {result}, rounds: {rounds}')

        result, _, rounds = play(p2, p1, verbose=False)
        print(f'episode: {p+1:>2}/{plays}, result: {result}, rounds: {rounds}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plays',
                        type=int, default=20, metavar='P',
                        help='Total number of plays')
    parser.add_argument('-a', '--agent',
                        choices=('dqn', 'ddqn', 'dddqn'), default='dddqn',
                        help='Agent type to test.')
    parser.add_argument('-w', '--weights',
                        default='./weights/dddqn2.h5',
                        help='Weights file path')

    args = parser.parse_args()
    play_tournament(args.plays, args.agent, args.weights)


if __name__ == '__main__':
    main()
