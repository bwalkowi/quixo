import os
from collections import deque
from typing import Tuple, Optional
from random import random, choice, sample

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from utils import (Mark, Action, Result, ALL_MOVES,
                   get_possible_moves, get_encoded_possible_moves,
                   STATE_SPACE_SIZE, MOVE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 training: bool = False,
                 batch_size: int = 32,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './weights/ddqn.h5') -> None:
        self.mark = mark
        self.training = training
        self.batch_size = batch_size

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.last_board = None
        self.last_move = None
        self.memory = deque(maxlen=2000)

        self.model = build_dqn(learning_rate)
        self.target_model = build_dqn(learning_rate)

        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)

    def move(self, board) -> Tuple[int, int, Action]:
        if random() < self.epsilon:
            possible_moves = get_possible_moves(board, self.mark)
            if possible_moves:
                chosen_move = choice(possible_moves)
                move_hash = ALL_MOVES.index(chosen_move)
            else:
                move_hash = 0
                chosen_move = ALL_MOVES[move_hash]
        else:
            predictions = self.model.predict(encode_board(board, self.mark))[0]
            possible_moves_hashes = get_encoded_possible_moves(board, self.mark)
            move_hash = next((x for _, x in sort_predictions(predictions)
                              if x in possible_moves_hashes), 0)
            chosen_move = ALL_MOVES[move_hash]

        if self.training:
            if self.last_board:
                last_state = encode_board(self.last_board, self.mark, as_batch=False)
                self.memory.append((last_state, self.last_move, 0, board, False))

            self.last_board = board
            self.last_move = move_hash

        return chosen_move

    def train(self, result: Result) -> None:
        last_state = encode_board(self.last_board, self.mark, as_batch=False)
        self.memory.append((last_state, self.last_move, result.value, None, True))

        self.last_move = None
        self.last_board = None
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) > 1000:
            batch = sample(self.memory, self.batch_size)

            states = np.array([state for state, *_ in batch])
            targets = self.model.predict(states)
            for i, (_, move_hash, reward, next_board, done) in enumerate(batch):
                if done:
                    targets[i, move_hash] = reward
                else:
                    next_state = encode_board(next_board, self.mark)
                    x = self.target_model.predict(next_state)[0]
                    x_valid = mask_invalid_moves(next_board, self.mark, x)
                    targets[i, move_hash] = reward + self.gamma * np.amax(x_valid)

            self.model.fit(states, targets, epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)
        self.target_model.load_weights(file_path)


def build_dqn(learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=STATE_SPACE_SIZE, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(MOVE_SPACE_SIZE, activation='linear'),
    ])
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=tf.losses.huber_loss)
    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(MOVE_SPACE_SIZE)), reverse=True)


def mask_invalid_moves(board, mark, predictions):
    opponent_mark = mark.opposite_mark()
    mask = [board[row][col] is opponent_mark for row, col, _ in ALL_MOVES]
    return np.ma.masked_array(predictions, mask=mask)
