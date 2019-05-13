import random
from typing import Tuple
from copy import deepcopy

from utils import (Mark, Action, BOARD_SIZE,
                   get_possible_moves, apply_move)


class Player:
    def __init__(self, mark: Mark, *,
                 eval_rows: bool = True,
                 eval_cols: bool = False,
                 eval_diagonals: bool = False) -> None:
        self.mark = mark
        self.eval_rows = eval_rows
        self.eval_cols = eval_cols
        self.eval_diagonals = eval_diagonals

    def move(self, board) -> Tuple[int, int, Action]:
        possible_moves = get_possible_moves(board, self.mark)
        possible_moves.sort(key=lambda m: (evaluate_move(board, self.mark, m,
                                                         self.eval_rows,
                                                         self.eval_cols,
                                                         self.eval_diagonals),
                                           random.random()))
        return possible_moves[-1]


def evaluate_move(board,
                  mark: Mark,
                  move: Tuple[int, int, Action],
                  eval_rows: bool = True,
                  eval_cols: bool = False,
                  eval_diagonals: bool = False) -> int:

    row, col, action = move
    next_board = deepcopy(board)
    apply_move(next_board, row, col, action, mark)

    score = 0
    if eval_rows:
        score += sum(row.count(mark)**3 for row in next_board)
    if eval_cols:
        score += sum(sum(row[col] is mark for row in next_board)**3
                     for col in range(BOARD_SIZE))
    if eval_diagonals:
        score += sum(next_board[i][i] is mark for i in range(BOARD_SIZE))**2
        score += sum(next_board[i][BOARD_SIZE-1-i] is mark
                     for i in range(BOARD_SIZE))**2

    return score
