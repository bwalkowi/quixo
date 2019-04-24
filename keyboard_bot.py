from typing import Tuple
from termcolor import colored

from utils import Mark, Action


ACTIONS = '[0-PUSH_DOWN/1-PUSH_UP/2-PUSH_RIGHT/3-PUSH_LEFT]'
PROMPT = f'Enter move (Row[0-4] Col[0-4] Action{ACTIONS}): '


class Player:
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def move(self, board) -> Tuple[int, int, Action]:
        try:
            row, col, action = input(PROMPT).split()
            return int(row), int(col), Action(int(action))
        except Exception as ex:
            print(colored('[ERROR]', 'red'), ex)
            return self.move(board)
