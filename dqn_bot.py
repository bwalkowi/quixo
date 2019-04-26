import os
from random import random, choice
from typing import Tuple, Optional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from utils import (Mark, Action, Result, ALL_MOVES, get_possible_moves,
                   get_encoded_possible_moves, get_encoded_impossible_moves,
                   STATE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 training: bool = False,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './qwe.h5') -> None:
        self.mark = mark
        self.training = training

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.buffer = []
        self.model = build_dqn(learning_rate)

        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)

    def move(self, board) -> Tuple[int, int, Action]:
        mark = self.mark

        if self.training:
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
                encoded_move = next((x for _, x in sort_predictions(predictions)
                                     if x in encoded_possible_moves), 0)
                chosen_move = ALL_MOVES[encoded_move]

            self.buffer.append((board, encoded_move))
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board, mark))[0]
            encoded_possible_moves = get_encoded_possible_moves(board, mark)
            encoded_move = next((x for _, x in sort_predictions(predictions)
                                 if x in encoded_possible_moves), 0)
            return ALL_MOVES[encoded_move]

    def train(self, result: Result) -> None:
        batch = []
        moves = []
        qs = []

        reward = result.value
        if result is Result.DISQUALIFIED:
            last_board, last_move = self.buffer[-1]
            batch.append(encode_board(last_board, self.mark, as_batch=False))
            moves.append(last_move)
            qs.append(reward)
        else:
            for board, encoded_move in self.buffer[::-1]:
                batch.append(encode_board(board, self.mark, as_batch=False))

                state_moves = get_encoded_impossible_moves(board, self.mark)
                state_moves.append(encoded_move)
                moves.append(state_moves)

                state_qs = [Result.DISQUALIFIED.value] * len(state_moves)
                state_qs[-1] = reward
                qs.append(state_qs)

                reward *= self.gamma

        batch = np.array(batch)
        targets = self.model.predict(batch)
        for i, (state_moves, state_qs) in enumerate(zip(moves, qs)):
            targets[i, state_moves] = state_qs

        self.model.fit(batch, targets, epochs=1, verbose=0)

        self.buffer = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)


def build_dqn(learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=STATE_SPACE_SIZE, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(ALL_MOVES), activation='linear'),
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(len(ALL_MOVES))), reverse=True)
