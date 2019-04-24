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
    WIN = 10
    LOSS = -10
    DRAW = 0
    DISQUALIFIED = -1000

    def __str__(self) -> str:
        return self.name


ALL_MOVES = [
    (0, 0, Action.PUSH_UP),
    (0, 0, Action.PUSH_LEFT),
    (0, 1, Action.PUSH_UP),
    (0, 1, Action.PUSH_LEFT),
    (0, 1, Action.PUSH_RIGHT),
    (0, 2, Action.PUSH_UP),
    (0, 2, Action.PUSH_LEFT),
    (0, 2, Action.PUSH_RIGHT),
    (0, 3, Action.PUSH_UP),
    (0, 3, Action.PUSH_LEFT),
    (0, 3, Action.PUSH_RIGHT),
    (0, 4, Action.PUSH_UP),
    (0, 4, Action.PUSH_RIGHT),

    (1, 0, Action.PUSH_UP),
    (1, 0, Action.PUSH_DOWN),
    (1, 0, Action.PUSH_LEFT),
    (1, 4, Action.PUSH_UP),
    (1, 4, Action.PUSH_DOWN),
    (1, 4, Action.PUSH_RIGHT),

    (2, 0, Action.PUSH_UP),
    (2, 0, Action.PUSH_DOWN),
    (2, 0, Action.PUSH_LEFT),
    (2, 4, Action.PUSH_UP),
    (2, 4, Action.PUSH_DOWN),
    (2, 4, Action.PUSH_RIGHT),

    (3, 0, Action.PUSH_UP),
    (3, 0, Action.PUSH_DOWN),
    (3, 0, Action.PUSH_LEFT),
    (3, 4, Action.PUSH_UP),
    (3, 4, Action.PUSH_DOWN),
    (3, 4, Action.PUSH_RIGHT),

    (4, 0, Action.PUSH_DOWN),
    (4, 0, Action.PUSH_LEFT),
    (4, 1, Action.PUSH_DOWN),
    (4, 1, Action.PUSH_LEFT),
    (4, 1, Action.PUSH_RIGHT),
    (4, 2, Action.PUSH_DOWN),
    (4, 2, Action.PUSH_LEFT),
    (4, 2, Action.PUSH_RIGHT),
    (4, 3, Action.PUSH_DOWN),
    (4, 3, Action.PUSH_LEFT),
    (4, 3, Action.PUSH_RIGHT),
    (4, 4, Action.PUSH_DOWN),
    (4, 4, Action.PUSH_RIGHT)
]
STATE_SPACE_SIZE = 50


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
        for i in range(row, 4):
            board[i][col] = board[i+1][col]
        board[4][col] = mark
    elif action == Action.PUSH_LEFT:
        for i in range(col, 4):
            board[row][i] = board[row][i+1]
        board[row][4] = mark
    else:
        for i in range(col, 0, -1):
            board[row][i] = board[row][i-1]
        board[row][0] = mark


def get_winners(board) -> Set[Mark]:
    winners = set()

    # check horizontal
    for row in board:
        mark = row[0]
        if mark != Mark.EMPTY and row.count(mark) == 5:
            winners.add(mark)

    # check vertical
    for col in range(5):
        mark = board[0][col]
        if mark != Mark.EMPTY and all(board[r][col] == mark for r in range(5)):
            winners.add(mark)

    # check diagonals
    mark = board[2][2]
    if mark != Mark.EMPTY:
        if all(board[i][i] == mark for i in range(5)):
            winners.add(mark)
        if all(board[i][4-i] == mark for i in range(5)):
            winners.add(mark)

    return winners
