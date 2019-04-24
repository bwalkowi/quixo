from typing import Tuple
from random import choice

from utils import Mark, Action, ALL_MOVES, get_possible_moves


class Player:
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def move(self, board) -> Tuple[int, int, Action]:
        possible_moves = get_possible_moves(board, self.mark)
        return choice(possible_moves) if possible_moves else ALL_MOVES[0]
