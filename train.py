import argparse
from typing import Optional

from quixo import play
from utils import Mark, Result


def train(agent: str, src_path: Optional[str], dst_path: Optional[str]) -> None:
    if agent == 'dqn':
        from dqn_bot import Player
    elif agent == 'ddqn':
        from ddqn_bot import Player
    else:
        from dddqn_bot import Player

    reference_agent = Player(Mark.X, weights_file=src_path)
    learning_agent = Player(Mark.O, training=True, weights_file=src_path)

    play_no = 0
    while True:
        plays = 1
        while plays < 100:
            _, result, rounds = play(reference_agent, learning_agent, verbose=False)
            learning_agent.train(result)
            print(f'episode: {play_no}, score: {result.value}, rounds: {rounds}')
            play_no += 1
            plays += 1

        plays = 1
        while plays < 50:
            result, _, rounds = play(learning_agent, reference_agent, verbose=False)
            learning_agent.train(result)
            print(f'episode: {play_no}, score: {result.value}, rounds: {rounds}')
            play_no += 1
            plays += 1

        reference_agent.model.set_weights(learning_agent.model.get_weights())
        if dst_path:
            learning_agent.save(dst_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent',
                        choices=('dqn', 'ddqn', 'dddqn'), default='dqn',
                        help='Agent type to teach.')
    parser.add_argument('-s', '--src',
                        help='File containing weights to load.')
    parser.add_argument('-d', '--dst',
                        help='File to which save learned weights.')

    args = parser.parse_args()
    train(args.agent, args.src, args.dst)


if __name__ == '__main__':
    main()
