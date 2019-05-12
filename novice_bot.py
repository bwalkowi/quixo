import random
from typing import Tuple
from copy import deepcopy

from utils import (Mark, Action, BOARD_SIZE,
                   get_possible_moves, apply_move)


class Player:
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def move(self, board) -> Tuple[int, int, Action]:
        possible_moves = get_possible_moves(board, self.mark)
        possible_moves.sort(key=lambda x: (evaluate_move(board, self.mark, x),
                                           random.random()),
                            reverse=True)
        return possible_moves[0]


def evaluate_move(board, mark: Mark, move: Tuple[int, int, Action]) -> int:
    row, col, action = move
    next_board = deepcopy(board)
    apply_move(next_board, row, col, action, mark)

    score = sum(row.count(mark)**3 for row in next_board)
    score += sum(sum(row[col] is mark for row in next_board)**3
                 for col in range(BOARD_SIZE))
    score += sum(next_board[i][i] is mark for i in range(BOARD_SIZE))**2
    score += sum(next_board[i][BOARD_SIZE-1-i] is mark
                 for i in range(BOARD_SIZE))**2

    return score
