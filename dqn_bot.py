import os
from random import random, choice
from typing import Tuple, Optional

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from utils import (Mark, Action, Result, ALL_MOVES, get_possible_moves,
                   get_encoded_possible_moves, get_encoded_invalid_moves,
                   STATE_SPACE_SIZE, MOVE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 training: bool = False,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './weights/dqn.h5') -> None:
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
            move_hash = 0
            chosen_move = ALL_MOVES[move_hash]

            if random() < self.epsilon:
                possible_moves = get_possible_moves(board, mark)
                if possible_moves:
                    chosen_move = choice(possible_moves)
                    move_hash = ALL_MOVES.index(chosen_move)
            else:
                predictions = self.model.predict(encode_board(board, mark))[0]
                possible_moves_hashes = get_encoded_possible_moves(board, mark)
                move_hash = next((x for _, x in sort_predictions(predictions)
                                  if x in possible_moves_hashes), 0)
                chosen_move = ALL_MOVES[move_hash]

            self.buffer.append((board, move_hash))
            return chosen_move
        else:
            predictions = self.model.predict(encode_board(board, mark))[0]
            possible_moves_hashes = get_encoded_possible_moves(board, mark)
            move_hash = next((x for _, x in sort_predictions(predictions)
                              if x in possible_moves_hashes), 0)
            return ALL_MOVES[move_hash]

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
            for board, move_hash in self.buffer[::-1]:
                invalid_moves = get_encoded_invalid_moves(board, self.mark)
                invalid_moves_qs = [Result.DISQUALIFIED.value] * len(invalid_moves)

                batch.append(encode_board(board, self.mark, as_batch=False))
                moves.append([move_hash, *invalid_moves])
                qs.append([reward, *invalid_moves_qs])

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
        Dense(128, activation='relu'),
        Dense(MOVE_SPACE_SIZE, activation='linear'),
    ])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(MOVE_SPACE_SIZE)), reverse=True)
