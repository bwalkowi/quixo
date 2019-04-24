import os
import random as rand
from collections import deque
from typing import Tuple, Optional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from utils import (Mark, Action, Result,
                   ALL_MOVES, get_possible_moves,
                   STATE_SPACE_SIZE, encode_board)


BATCH_SIZE = 32


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
    rounds_played: int = 0

    def __init__(self, mark: Mark, *,
                 learning: bool = False,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './ddqnw.h5') -> None:
        self.mark = mark
        self.learning = learning

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.prev_state = None
        self.prev_action = None
        self.memory = deque(maxlen=2000)

        self.model = build_dqn(learning_rate)
        self.target_model = build_dqn(learning_rate)

        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)
        self.update_target_model()

    def move(self, board) -> Tuple[int, int, Action]:
        if self.learning:
            if self.prev_state:
                self.memory.append((self.prev_state,
                                    self.prev_action,
                                    0, board, False))
            encoded_move = 0
            chosen_move = ALL_MOVES[encoded_move]

            if rand.random() < self.epsilon:
                possible_moves = get_possible_moves(board, self.mark)
                if possible_moves:
                    chosen_move = rand.choice(possible_moves)
                    encoded_move = ALL_MOVES.index(chosen_move)
            else:
                predictions = self.model.predict(encode_board(board, self.mark))
                encoded_move = np.argmax(predictions[0])
                chosen_move = ALL_MOVES[encoded_move]

            self.prev_state = board
            self.prev_action = encoded_move
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board, self.mark))
            return ALL_MOVES[np.argmax(predictions[0])]

    def train(self, result: Result) -> None:
        self.memory.append((self.prev_state, self.prev_action,
                            result.value, None, True))
        self.prev_state = None
        self.prev_action = None
        self.rounds_played += 1

        if len(self.memory) < BATCH_SIZE or self.rounds_played % 4 != 0:
            return

        selected_plays = rand.sample(self.memory, BATCH_SIZE)
        batch = np.array([encode_board(x[0], self.mark, as_batch=False)
                          for x in selected_plays])
        targets = self.model.predict(batch)

        batch2 = np.array([encode_board(x[3], self.mark, as_batch=False)
                           for x in selected_plays if x[3]])
        targets2 = self.target_model.predict(batch2)

        j = 0
        for i, (_, action, reward, _, done) in enumerate(selected_plays):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + self.gamma * np.amax(targets2[j])
                j += 1

        self.model.fit(batch, targets, epochs=1, verbose=0)
        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
