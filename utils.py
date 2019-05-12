from enum import Enum
from typing import Set, List, Tuple

import numpy as np


class Mark(Enum):
    EMPTY = ' '
    X = 'X'
    O = 'O'

    def opposite_mark(self):
        return Mark.X if self.value == 'O' else Mark.O

    def __str__(self) -> str:
        return self.name


class Action(Enum):
    PUSH_DOWN = 0
    PUSH_UP = 1
    PUSH_RIGHT = 2
    PUSH_LEFT = 3

    def __str__(self) -> str:
        return self.name


class Result(Enum):
    WIN = 1
    LOSS = -1
    DRAW = 0
    DISQUALIFIED = 0

    def __str__(self) -> str:
        return self.name


BOARD_SIZE = 5
STATE_SPACE_SIZE = 2 * BOARD_SIZE**2

ALL_MOVES = [(0, 0, Action.PUSH_UP),
             (0, 0, Action.PUSH_LEFT)]
ALL_MOVES.extend([(0, col, action)
                  for col in range(1, BOARD_SIZE-1)
                  for action in (Action.PUSH_LEFT, Action.PUSH_UP,
                                 Action.PUSH_RIGHT)])
ALL_MOVES.extend([(0, BOARD_SIZE-1, Action.PUSH_UP),
                  (0, BOARD_SIZE-1, Action.PUSH_RIGHT)])

for r in range(1, BOARD_SIZE-1):
    ALL_MOVES.extend([(r, 0, Action.PUSH_UP),
                      (r, 0, Action.PUSH_DOWN),
                      (r, 0, Action.PUSH_LEFT),
                      (r, BOARD_SIZE - 1, Action.PUSH_UP),
                      (r, BOARD_SIZE - 1, Action.PUSH_DOWN),
                      (r, BOARD_SIZE - 1, Action.PUSH_RIGHT)])

ALL_MOVES.extend([(BOARD_SIZE-1, 0, Action.PUSH_DOWN),
                  (BOARD_SIZE-1, 0, Action.PUSH_LEFT)])
ALL_MOVES.extend([(BOARD_SIZE-1, col, action)
                  for col in range(1, BOARD_SIZE-1)
                  for action in (Action.PUSH_LEFT, Action.PUSH_DOWN,
                                 Action.PUSH_RIGHT)])
ALL_MOVES.extend([(BOARD_SIZE-1, BOARD_SIZE-1, Action.PUSH_DOWN),
                  (BOARD_SIZE-1, BOARD_SIZE-1, Action.PUSH_RIGHT)])

MOVE_SPACE_SIZE = len(ALL_MOVES)


def encode_board(board, mark: Mark, as_batch: bool = True) -> np.ndarray:
    opponent_mark = mark.opposite_mark()

    p1 = [cell is mark for row in board for cell in row]
    p2 = [cell is opponent_mark for row in board for cell in row]
    if as_batch:
        return np.array([p1 + p2], dtype=np.int32)
    else:
        return np.array(p1 + p2, dtype=np.int32)


def get_possible_moves(board, mark: Mark) -> List[Tuple[int, int, Action]]:
    return [(row, col, action) for row, col, action in ALL_MOVES
            if board[row][col] in (mark, Mark.EMPTY)]


def get_encoded_possible_moves(board, mark: Mark) -> List[int]:
    return [i for i, (row, col, _) in enumerate(ALL_MOVES)
            if board[row][col] in (mark, Mark.EMPTY)]


def get_encoded_invalid_moves(board, mark: Mark) -> List[int]:
    opposite_mark = mark.opposite_mark()
    return [i for i, (row, col, _) in enumerate(ALL_MOVES)
            if board[row][col] is opposite_mark]


def is_valid_move(board, mark: Mark, row: int, col: int, action: Action) -> bool:
    if board[row][col] in (mark, Mark.EMPTY):
        return (row, col, action) in ALL_MOVES
    else:
        return False


def apply_move(board, row: int, col: int, action: Action, mark: Mark) -> None:
    if action == Action.PUSH_DOWN:
        for i in range(row, 0, -1):
            board[i][col] = board[i-1][col]
        board[0][col] = mark
    elif action == Action.PUSH_UP:
        for i in range(row, BOARD_SIZE-1):
            board[i][col] = board[i+1][col]
        board[BOARD_SIZE-1][col] = mark
    elif action == Action.PUSH_LEFT:
        for i in range(col, BOARD_SIZE-1):
            board[row][i] = board[row][i+1]
        board[row][BOARD_SIZE-1] = mark
    else:
        for i in range(col, 0, -1):
            board[row][i] = board[row][i-1]
        board[row][0] = mark


def get_winners(board) -> Set[Mark]:
    winners = set()

    # check horizontal
    for row in board:
        mark = row[0]
        if mark != Mark.EMPTY and row.count(mark) is BOARD_SIZE:
            winners.add(mark)

    # check vertical
    for col in range(BOARD_SIZE):
        mark = board[0][col]
        if mark != Mark.EMPTY and all(board[row][col] is mark
                                      for row in range(BOARD_SIZE)):
            winners.add(mark)

    # check diagonals
    mark = board[0][0]
    if mark != mark.EMPTY and all(board[i][i] is mark
                                  for i in range(BOARD_SIZE)):
        winners.add(mark)

    mark = board[0][BOARD_SIZE-1]
    if mark != mark.EMPTY and all(board[i][BOARD_SIZE-1-i] is mark
                                  for i in range(BOARD_SIZE)):
        winners.add(mark)

    return winners
