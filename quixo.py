import sys
import inspect
import argparse
import traceback
from typing import Tuple
from copy import deepcopy
from termcolor import colored
from importlib import import_module

from utils import (Mark, Result, BOARD_SIZE,
                   is_valid_move, apply_move, get_winners)


ERROR = colored('[ERROR]', 'red')


def play_game(player1, player2, *,
              verbose: bool = True,
              max_rounds: int = 500) -> Tuple[Result, Result, int]:
    round_no = 0
    game_end = False
    players = (player1, player2)
    results = [Result.DRAW, Result.DRAW]
    board = [[Mark.EMPTY]*BOARD_SIZE for _ in range(BOARD_SIZE)]

    while round_no < max_rounds and not game_end:
        for i, player in enumerate(players):
            row, col, action = player.move(deepcopy(board))
            if not is_valid_move(board, player.mark, row, col, action):
                print(ERROR, f'Player {player.mark} disqualified for '
                             f'invalid move: {row}, {col}, {action}',
                      file=sys.stderr)
                results[i] = Result.DISQUALIFIED
                results[1-i] = Result.WIN
                game_end = True
                break

            apply_move(board, row, col, action, player.mark)
            if verbose:
                print(f'Player {player.mark} makes move: {row}, {col}, {action}\n',
                      '      ', '----' * BOARD_SIZE, '-', sep='')
                for row in board:
                    print('      | ', ' | '.join(c.value for c in row), ' |\n',
                          '      ', '----' * BOARD_SIZE, '-', sep='')

            winners = get_winners(board)
            if not winners:
                continue
            elif len(winners) == 2:
                game_end = True
                break
            elif winners.pop() == player.mark:
                game_end = True
                results[i] = Result.WIN
                results[1-i] = Result.LOSS
                break
            else:
                game_end = True
                results[i] = Result.LOSS
                results[1-i] = Result.WIN
                break

        round_no += 1

    p1_result, p2_result = results
    return p1_result, p2_result, round_no


def play_tournament(player1, player2, *,
                    plays: int,
                    verbose: bool,
                    max_rounds: int) -> None:
    p1 = player1
    p2 = player2
    for episode in range(1, plays+1):
        res1, res2, rounds = play_game(p1, p2, verbose=verbose,
                                       max_rounds=max_rounds)
        p1, p2 = p2, p1
        print(f'episode: {episode:>2}/{plays}, '
              f'result: {res1 if episode % 2 == 0 else res2}, '
              f'rounds: {rounds}')


def create_player(mod: str, mark: Mark, **kw_args):
    mod = import_module(mod)
    sig = inspect.signature(mod.Player)
    parsed_kw_args = {key: sig.parameters[key].annotation(val)
                      for key, val in kw_args.items()}

    return mod.Player(mark, **parsed_kw_args)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', help='Bot playing O')
    parser.add_argument('player2', help='Bot playing X')
    parser.add_argument('--args1',
                        metavar=('KEY', 'VALUE'),
                        action='append', nargs=2, default=[],
                        help='Arguments used for initializing player1')
    parser.add_argument('--args2',
                        metavar=('KEY', 'VALUE'),
                        action='append', nargs=2, default=[],
                        help='Arguments used for initializing player2')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='If specified moves and board will be printed '
                             'after each turn')
    parser.add_argument('-p', '--plays',
                        metavar='P',
                        type=int, default=1,
                        help='Number of games that should be played.')
    parser.add_argument('-r', '--max-rounds',
                        metavar='R',
                        type=int, default=100,
                        help='Maximum number of rounds after which '
                             'game ends in a draw.')

    args = parser.parse_args()
    try:
        player1 = create_player(args.player1, Mark.O, **dict(args.args1))
        player2 = create_player(args.player2, Mark.X, **dict(args.args2))
        play_tournament(player1, player2,
                        plays=args.plays,
                        verbose=args.verbose,
                        max_rounds=args.max_rounds)
    except (ModuleNotFoundError, AttributeError) as ex:
        print(ERROR, ex, file=sys.stderr)
        exit(1)
    except KeyboardInterrupt:
        print('\n', ERROR, 'Game interrupted!!', file=sys.stderr)
    except Exception as ex:
        print(ERROR, ex, file=sys.stderr)
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
