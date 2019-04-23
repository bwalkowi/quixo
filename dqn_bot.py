import random as rand
from typing import Tuple

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from utils import Mark, Action, Result, ALL_MOVES, get_possible_moves


STATE_SIZE = 50
ACTION_SIZE = 44


def build_dqn(learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=STATE_SIZE, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(ACTION_SIZE, activation='linear'),
    ])
    model.compile(optimizer=SGD(lr=learning_rate), loss='mse')

    return model


def encode_board(board, batch: bool = True):
    os = [cell is Mark.O for row in board for cell in row]
    xs = [cell is Mark.X for row in board for cell in row]
    if batch:
        return np.array([os + xs], dtype=np.int32)
    else:
        return np.array(os + xs, dtype=np.int32)


class Player:
    def __init__(self, mark: Mark, *,
                 train: bool = False,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001) -> None:
        self.mark = mark
        self.train = train

        self.buffer = []
        self.model = build_dqn(learning_rate)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def end_game(self, result: Result) -> None:
        if self.train:
            reward = result.value
            last_state, last_action = self.buffer[-1]

            batch = [encode_board(last_state, batch=False)]
            target_qs = [(last_action, reward)]

            if result is not Result.DISQUALIFIED:
                for state, action in self.buffer[-2::-1]:
                    reward *= self.gamma
                    batch.append(encode_board(state, batch=False))
                    target_qs.append((action, reward))

            batch = np.array(batch)
            targets = self.model.predict(batch)
            for i, (action, q) in enumerate(target_qs):
                targets[i, action] = q

            self.model.fit(batch, targets, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def move(self, board) -> Tuple[int, int, Action]:
        if self.train:
            encoded_move = 0
            chosen_move = ALL_MOVES[encoded_move]

            if rand.random() < self.epsilon:
                possible_moves = get_possible_moves(board, self.mark)
                if possible_moves:
                    chosen_move = rand.choice(possible_moves)
                    encoded_move = ALL_MOVES.index(chosen_move)
            else:
                predictions = self.model.predict(encode_board(board))
                encoded_move = np.argmax(predictions[0])
                chosen_move = ALL_MOVES[encoded_move]

            self.buffer.append((board, encoded_move))
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board))
            return ALL_MOVES[np.argmax(predictions[0])]

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)