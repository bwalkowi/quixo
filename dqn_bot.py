import os
from random import random, choice
from typing import Tuple, Optional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from utils import (Mark, Action, Result,
                   ALL_MOVES, get_possible_moves,
                   STATE_SPACE_SIZE, encode_board)


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
        self.model = build_dqn(learning_rate)

        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)

    def move(self, board) -> Tuple[int, int, Action]:
        if self.learning:
            encoded_move = 0
            chosen_move = ALL_MOVES[encoded_move]

            if random() < self.epsilon:
                possible_moves = get_possible_moves(board, self.mark)
                if possible_moves:
                    chosen_move = choice(possible_moves)
                    encoded_move = ALL_MOVES.index(chosen_move)
            else:
                predictions = self.model.predict(encode_board(board,
                                                              self.mark))
                encoded_move = np.argmax(predictions[0])
                chosen_move = ALL_MOVES[encoded_move]

            self.buffer.append((board, encoded_move))
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board, self.mark))
            return ALL_MOVES[np.argmax(predictions[0])]

    def train(self, result: Result) -> None:
        reward = result.value
        last_state, last_action = self.buffer[-1]

        batch = [encode_board(last_state, self.mark, as_batch=False)]
        target_qs = [(last_action, reward)]

        if result is not Result.DISQUALIFIED:
            for state, action in self.buffer[-2::-1]:
                reward *= self.gamma
                batch.append(encode_board(state, self.mark,
                                          as_batch=False))
                target_qs.append((action, reward))

        batch = np.array(batch)
        targets = self.model.predict(batch)
        for i, (action, q) in enumerate(target_qs):
            targets[i, action] = q

        self.model.fit(batch, targets, epochs=1, verbose=0)

        self.buffer = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)
