import argparse
import traceback
from copy import deepcopy
from termcolor import colored
from importlib import import_module

from utils import Mark, Result, is_valid_move, apply_move, get_winners


def play(player1, player2, verbose: bool = True, limit: int = 100) -> None:
    round_no = 0
    game_end = False
    players = (player1, player2)
    results = [Result.DRAW, Result.DRAW]
    board = [[Mark.EMPTY for _ in range(5)] for _ in range(5)]

    while round_no < limit and not game_end:
        for i, player in enumerate(players):
            row, col, action = player.move(deepcopy(board))
            if not is_valid_move(board, player.mark, row, col, action):
                print(f'{colored("[ERROR]", "red")} Player {player.mark} '
                      f'disqualified for invalid move: {row}, {col}, {action}')
                results[i] = Result.LOSS
                results[1-i] = Result.WIN
                game_end = True
                break

            print(f'Player {player.mark} makes move: {row}, {col}, {action}\n')
            apply_move(board, row, col, action, player.mark)
            if verbose:
                print_board(board)

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

    print('\nGame ends with results:')
    for player, result in zip(players, results):
        print(f'\tPlayer {player.mark} {result}')
        player.end_game(result)


def print_board(board) -> None:
    print('      ---------------------')
    for row in board:
        print('      |', ' | '.join(cell.value for cell in row), '|')
        print('      ---------------------')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('player1', help='Bot playing O')
    parser.add_argument('player2', help='Bot playing X')
    parser.add_argument('-v', '--verbose',
                        help='If specified board will be printed '
                             'after each turn',
                        action='store_true')
    parser.add_argument('-l', '--limit',
                        type=int, default=100,
                        help='Rounds limit')

    args = parser.parse_args()
    try:
        player1 = import_module(args.player1).Player(Mark.O)
        player2 = import_module(args.player2).Player(Mark.X)
        play(player1, player2, args.verbose, args.limit)
    except (ModuleNotFoundError, AttributeError) as ex:
        print(colored('[ERROR]', 'red'), ex)
        exit(1)
    except KeyboardInterrupt:
        print('\n', colored('[ERROR]', 'red'), 'Game abruptly ended!!')
    except Exception as ex:
        print(colored('[ERROR]', 'red'), ex)
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
