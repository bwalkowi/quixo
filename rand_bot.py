from typing import Tuple
from random import choice

from utils import Mark, Action, Result, get_possible_moves


class Player:
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def end_game(self, _result: Result) -> None:
        pass

    def move(self, board) -> Tuple[int, int, Action]:
        possible_moves = get_possible_moves(board, self.mark)
        if possible_moves:
            return choice(possible_moves)
        else:
            return 0, 0, Action.PUSH_UP
