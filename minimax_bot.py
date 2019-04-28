import random
from enum import Enum
from collections import OrderedDict
from typing import NamedTuple, Tuple, Optional

from utils import Mark, Action, get_possible_moves, apply_move, get_winners


MAX_SCORE = 1000


class Player:

    def __init__(self, mark: Mark, max_depth: int = 5) -> None:
        self.mark = mark
        self.max_depth = max_depth

    def move(self, board) -> Tuple[int, int, Action]:
        for depth in range(1, self.max_depth, 2):
            minimax(board, self.mark, depth)
        return minimax(board, self.mark, self.max_depth)


def minimax(board, mark: Mark, depth: int) -> Tuple[int, int, Action]:
    game = Quixo(board, mark, depth-1)
    best_move = 0, 0, Action.PUSH_UP
    opponent_mark = mark.opposite_mark()
    alpha, beta = float('-inf'), float('inf')

    for move, next_game in game.get_ordered_next_games(mark, alpha, beta):
        score = min_score(next_game, opponent_mark, depth-1, alpha, beta)
        if score > alpha:
            alpha = score
            best_move = move

            if alpha == MAX_SCORE:
                break

    return best_move


def max_score(game: 'Quixo', mark: Mark, depth: int,
              alpha: float, beta: float) -> float:

    opponent_mark = mark.opposite_mark()
    best_score = game.maybe_evaluate(mark, depth, alpha, beta)
    if best_score is None:
        best_score = float('-inf')
        for _, next_game in game.get_ordered_next_games(mark, alpha, beta):
            score = min_score(next_game, opponent_mark, depth-1, alpha, beta)
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        game.cache.put(game.hash(mark), best_score, depth, alpha, beta)
    return best_score


def min_score(game: 'Quixo', mark: Mark, depth: int,
              alpha: float, beta: float) -> float:

    opponent_mark = mark.opposite_mark()
    best_score = game.maybe_evaluate(mark, depth, alpha, beta)
    if best_score is None:
        best_score = float('inf')
        for _, next_game in game.get_ordered_next_games(mark, alpha,
                                                        beta, desc=False):
            score = max_score(next_game, opponent_mark, depth-1, alpha, beta)
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break

        game.cache.put(game.hash(mark), best_score, depth, alpha, beta)
    return best_score


class CacheEntryState(Enum):
    EXACT = 0
    UPPER_BOUND = 1
    LOWER_BOUND = 2


class CacheEntry(NamedTuple):
    depth: int
    score: float
    state: CacheEntryState


class Cache:
    def __init__(self, max_items: int = 10**9) -> None:
        self._max_items = max_items
        self._cache = OrderedDict()

    def put(self, key: str, score: float, depth: int,
            alpha: float, beta: float) -> None:
        # when search was stopped due to reaching depth or analyzing all
        # possible moves exact score is known
        if depth == 0 or alpha < score < beta:
            state = CacheEntryState.EXACT

        # when max_score stop searching after surpassing minimizing player
        # limit only lower_bound of possible scores is known
        elif score >= beta:
            state = CacheEntryState.LOWER_BOUND
            score = beta

        # when min_score stop searching after surpassing maximizing player
        # limit only upper_bound of possible scores is known
        elif score <= alpha:
            state = CacheEntryState.UPPER_BOUND
            score = alpha

        self._cache[key] = CacheEntry(depth, score, state)

        if len(self._cache) > self._max_items:
            self._cache.popitem(last=False)

    def lookup(self, key: str, depth: int, alpha: float,
               beta: float) -> Optional[float]:
        entry = self._cache.get(key, None)

        if entry and entry.depth >= depth:
            if entry.state is CacheEntryState.EXACT:
                return entry.score

            # if search window boundaries does not exceed the ones when
            # previous search was stopped then saved score bound can be
            # returned; otherwise search must be conducted once again
            elif entry.state is CacheEntryState.LOWER_BOUND and entry.score >= beta:
                return entry.score
            elif entry.state is CacheEntryState.UPPER_BOUND and entry.score <= alpha:
                return entry.score

        return None


class Quixo:
    cache: Cache = Cache()

    def __init__(self, board, mark: Mark, max_depth: int) -> None:
        self.board = board
        self.initial_mark = mark
        self.max_depth = max_depth

    def hash(self, mark: Mark) -> str:
        board_hash = ''.join(self.board[i][j].value
                             for i in range(5)
                             for j in range(5))
        return mark.value + board_hash

    def move(self, row: int, col: int, action: Action, mark: Mark) -> 'Quixo':
        new_game = Quixo([row[:] for row in self.board],
                         self.initial_mark, self.max_depth)
        apply_move(new_game.board, row, col, action, mark)
        return new_game

    def maybe_evaluate(self, mark: Mark, depth: int, alpha: float, beta: float,
                       max_depth: Optional[int] = None) -> Optional[float]:
        score = self.cache.lookup(self.hash(mark), depth, alpha, beta)
        if score:
            return score

        winners = get_winners(self.board)

        # do not evaluate board if there are no winners
        # and depth limit is not yet reached
        if len(winners) == 0 and depth > 0:
            return None

        if max_depth is None:
            max_depth = self.max_depth

        # heuristic to evaluate board value
        if len(winners) == 0:
            board = self.board
            p1_mark = self.initial_mark

            score = sum(row.count(p1_mark)**3 for row in board)
            score += sum(sum(row[col] == p1_mark for row in board)**3 for col in range(5))
            score += sum(board[i][i] == p1_mark for i in range(5))**2
            score += sum(board[i][4-i] == p1_mark for i in range(5))**2

        elif len(winners) == 2:
            score = 0
        elif winners.pop() == self.initial_mark:
            score = MAX_SCORE - (max_depth - depth)
        else:
            score = (max_depth - depth) - MAX_SCORE

        self.cache.put(self.hash(mark), score, depth, alpha, beta)
        return score

    def get_ordered_next_games(self, mark: Mark, alpha: float, beta: float,
                               desc: bool = True):
        possible_games = [(move, self.move(*move, mark))
                          for move in get_possible_moves(self.board, mark)]
        # add some random noise to randomize results
        possible_games.sort(key=lambda x: (x[1].maybe_evaluate(mark, 0, alpha, beta, 0),
                                           random.random()),
                            reverse=desc)
        return possible_games
