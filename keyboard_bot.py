from typing import Tuple
from termcolor import colored

from utils import Mark, Action, Result


PROMPT = '''
Enter selected move in format: Row Col Action
Row: 0-4
Col: 0-4
Action: 0 (PUSH_DOWN) / 1 (PUSH_UP) / 2 (PUSH_RIGHT) / 3 (PUSH_LEFT)

Move: '''


class Player:
    def __init__(self, mark: Mark) -> None:
        self.mark = mark

    def end_game(self, _result: Result) -> None:
        pass

    def move(self, board) -> Tuple[int, int, Action]:
        try:
            row, col, action = input(PROMPT).split()
            row = int(row)
            col = int(col)
            action = Action(int(action))
        except Exception as ex:
            print(colored('[ERROR]', 'red'), ex)
            return self.move(board)
        else:
            return row, col, action
