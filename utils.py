from enum import Enum
from typing import Set, List, Tuple


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
    WIN = 0
    LOSS = 1
    DRAW = 2

    def __str__(self) -> str:
        return self.name


ALL_MOVES = {
    0: {
        0: (Action.PUSH_UP, Action.PUSH_LEFT),
        1: (Action.PUSH_UP, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        2: (Action.PUSH_UP, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        3: (Action.PUSH_UP, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        4: (Action.PUSH_UP, Action.PUSH_RIGHT),
    },
    1: {
        0: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_LEFT),
        4: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_RIGHT),
    },
    2: {
        0: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_LEFT),
        4: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_RIGHT),
    },
    3: {
        0: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_LEFT),
        4: (Action.PUSH_UP, Action.PUSH_DOWN, Action.PUSH_RIGHT),
    },
    4: {
        0: (Action.PUSH_DOWN, Action.PUSH_LEFT),
        1: (Action.PUSH_DOWN, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        2: (Action.PUSH_DOWN, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        3: (Action.PUSH_DOWN, Action.PUSH_LEFT, Action.PUSH_RIGHT),
        4: (Action.PUSH_DOWN, Action.PUSH_RIGHT)
    }
}


def get_possible_moves(board, mark: Mark) -> List[Tuple[int, int, Action]]:
    possible_moves = []
    for row, col_to_actions in ALL_MOVES.items():
        for col, actions in col_to_actions.items():
            if board[row][col] in (mark, Mark.EMPTY):
                possible_moves.extend((row, col, a) for a in actions)
    return possible_moves


def is_valid_move(board, mark: Mark, row: int, col: int, action: Action) -> bool:
    if board[row][col] in (mark, Mark.EMPTY):
        return action in ALL_MOVES.get(row, {}).get(col, ())
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
