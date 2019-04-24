import os
from random import random, choice
from typing import Tuple, Optional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from utils import (Mark, Action, Result, ALL_MOVES,
                   get_possible_moves, get_encoded_possible_moves,
                   STATE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 learning: bool = False,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './dqnw.h5') -> None:
        self.mark = mark
        self.learning = learning

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.buffer = []
        self.invalid_plays = []
        self.model = build_dqn(learning_rate)

        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)

    def move(self, board) -> Tuple[int, int, Action]:
        mark = self.mark

        if self.learning:
            encoded_move = 0
            chosen_move = ALL_MOVES[encoded_move]

            if random() < self.epsilon:
                possible_moves = get_possible_moves(board, mark)
                if possible_moves:
                    chosen_move = choice(possible_moves)
                    encoded_move = ALL_MOVES.index(chosen_move)
            else:
                predictions = self.model.predict(encode_board(board, mark))[0]
                encoded_possible_moves = get_encoded_possible_moves(board, mark)
                for _, encoded_move in sort_predictions(predictions):
                    if encoded_move in encoded_possible_moves:
                        break
                    else:
                        self.invalid_plays.append((board, encoded_move))

                chosen_move = ALL_MOVES[encoded_move]

            self.buffer.append((board, encoded_move))
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board, mark))[0]
            encoded_possible_moves = get_encoded_possible_moves(board, mark)
            for _, encoded_move in sort_predictions(predictions):
                if encoded_move in encoded_possible_moves:
                    return ALL_MOVES[encoded_move]
            return ALL_MOVES[0]

    def train(self, result: Result) -> None:
        batch = [encode_board(board, self.mark, as_batch=False)
                 for board, _ in self.invalid_plays]
        target_qs = [(action, Result.DISQUALIFIED.value)
                     for _, action in self.invalid_plays]

        reward = result.value
        if result is Result.DISQUALIFIED:
            last_board, last_action = self.buffer[-1]
            batch.append(encode_board(last_board, self.mark, as_batch=False))
            target_qs.append((last_action, reward))
        else:
            for state, action in self.buffer[::-1]:
                batch.append(encode_board(state, self.mark, as_batch=False))
                target_qs.append((action, reward))
                reward *= self.gamma

        batch = np.array(batch)
        targets = self.model.predict(batch)
        for i, (action, q) in enumerate(target_qs):
            targets[i, action] = q

        self.model.fit(batch, targets, epochs=1, verbose=0)

        self.buffer = []
        self.invalid_plays = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)


def build_dqn(learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=STATE_SPACE_SIZE,
              activation='relu',
              kernel_initializer='zeros'),
        Dense(128, activation='relu',
              kernel_initializer='zeros'),
        Dense(64, activation='relu',
              kernel_initializer='zeros'),
        Dense(len(ALL_MOVES),
              activation='linear',
              kernel_initializer='zeros'),
    ])
    model.compile(optimizer=SGD(lr=learning_rate), loss='mse')

    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(len(ALL_MOVES))), reverse=True)
