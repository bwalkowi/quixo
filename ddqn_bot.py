import os
from collections import deque
from typing import Tuple, Optional
from random import random, choice, sample

import numpy as np
from numpy.ma import masked_array
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import RMSprop

from utils import (Mark, Action, Result, ALL_MOVES,
                   get_possible_moves, get_encoded_possible_moves,
                   STATE_SPACE_SIZE, MOVE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 training: bool = False,
                 batch_size: int = 64,
                 max_tau: int = 250,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.1,
                 epsilon_decay: float = 0.99,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './weights/ddqn.h5') -> None:
        self.mark = mark
        self.training = training
        self.batch_size = batch_size

        self.tau = 1
        self.max_tau = max_tau

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.prev_board = None
        self.prev_move = None
        self.memory = deque(maxlen=10_000)

        self.model = build_model(learning_rate)
        self.target_model = build_model(learning_rate)

        # plot_model(self.model, to_file='ddqn_model.png',
        #            show_shapes=True, show_layer_names=True)
        # print(self.model.summary())

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
            self.tau += 1
            if self.tau > self.max_tau:
                self.tau = 0
                self.update_target_model()

            self.memorize(board, move_hash, 0, False)

        return chosen_move

    def train(self, result: Result) -> None:
        self.memorize(None, None, result.value, True)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) > 1000:
            rows = np.arange(self.batch_size)
            batch = sample(self.memory, self.batch_size)

            states, moves, rewards, next_states, masks, done = zip(*batch)

            states = np.array(states)
            q_targets = self.model.predict(states)

            next_states = np.array(next_states)
            next_qs_model = masked_array(self.model.predict(next_states),
                                         mask=np.array(masks))
            next_qs_target = self.target_model.predict(next_states)
            next_qs_values = next_qs_target[rows, np.argmax(next_qs_model, axis=1)]

            rewards = np.array(rewards)
            done = np.array(done, dtype=np.int32)

            q_targets[rows, moves] = rewards + self.gamma * next_qs_values * (1 - done)

            self.model.fit(states, q_targets, epochs=1, verbose=0)

    def memorize(self, board, move_hash, reward, done):
        if self.prev_board:
            prev_state = encode_board(self.prev_board, self.mark,
                                      as_batch=False)
            if done:
                invalid_moves_mask = [0] * MOVE_SPACE_SIZE
                curr_state = prev_state
            else:
                opponent_mark = self.mark.opposite_mark()
                invalid_moves_mask = [board[row][col] is opponent_mark
                                      for row, col, _ in ALL_MOVES]
                curr_state = encode_board(board, self.mark, as_batch=False)

            self.memory.append((prev_state, self.prev_move, reward,
                                curr_state, invalid_moves_mask, done))

        self.prev_board = board
        self.prev_move = move_hash

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, file_path: str) -> None:
        self.model.save_weights(file_path)

    def load(self, file_path: str) -> None:
        self.model.load_weights(file_path)
        self.target_model.load_weights(file_path)


def build_model(learning_rate: float = 0.001) -> Sequential:
    model = Sequential([
        Dense(128, input_dim=STATE_SPACE_SIZE, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(MOVE_SPACE_SIZE, activation='linear'),
    ])
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=tf.losses.huber_loss)
    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(MOVE_SPACE_SIZE)), reverse=True)
