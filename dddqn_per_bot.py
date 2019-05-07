import os
from typing import Tuple, Optional
from random import random, choice

import numpy as np
from numpy.ma import masked_array
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.layers import Input, Dense, Lambda

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
                 weights_file: Optional[str] = './weights/dddqn_per.h5') -> None:
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
        self.memory = Memory(capacity=10_000)

        self.model = build_model(learning_rate)
        self.target_model = build_model(learning_rate)

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
            samples = self.memory.sample(self.batch_size)
            states, moves, rewards, next_states, masks, done, indices, weights = samples

            q_targets = self.model.predict(states)
            next_qs_model = masked_array(self.model.predict(next_states), mask=masks)
            next_qs_target = self.target_model.predict(next_states)
            next_qs_values = next_qs_target[rows, np.argmax(next_qs_model, axis=1)]

            old_q_targets = q_targets[rows, moves]
            new_q_targets = rewards + self.gamma * next_qs_values * (1 - done)

            q_targets[rows, moves] = new_q_targets
            self.model.fit(states, q_targets, epochs=1, verbose=0)

            errors = np.abs(old_q_targets - new_q_targets)
            self.memory.batch_update(indices, errors)

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

            self.memory.add((prev_state, self.prev_move, reward,
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


class Memory:
    e: float = 0.01
    a: float = 0.6
    beta: float = 0.4
    beta_increment_per_sampling: float = 0.001

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity

        self.pos = 0
        self.entries_num = 0
        self.tree = np.zeros(2 * capacity - 1)  # sum tree
        self.data = np.zeros(capacity, dtype=object)

    def add(self, experience) -> None:
        self.data[self.pos] = experience
        self._update(self.pos + self.capacity - 1, 1.0 ** self.a)
        self.pos = (self.pos + 1) % self.capacity
        if self.entries_num < self.capacity:
            self.entries_num += 1

    def sample(self, batch_size: int):
        start = self.capacity - 1
        stop = self.entries_num + self.capacity - 1
        probabilities = self.tree[start:stop] / self.tree[0]

        indices = np.random.choice(self.entries_num, batch_size, p=probabilities)
        states, moves, rewards, next_states, masks, done = zip(*self.data[indices])

        states = np.array(states)
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        done = np.array(done, dtype=np.int32)

        weights = (self.entries_num * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        return states, moves, rewards, next_states, masks, done, indices, weights

    def batch_update(self, indices: np.ndarray, errors: np.ndarray) -> None:
        priorities = np.minimum(errors + self.e, 1.0) ** self.a
        for idx, priority in zip(indices, priorities):
            self._update(idx + self.capacity - 1, priority)

    def _update(self, idx: int, priority: float) -> None:
        diff = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += diff

    def __len__(self):
        return self.entries_num


def build_model(learning_rate: float = 0.001) -> Model:
    model_input = Input(shape=(STATE_SPACE_SIZE,), name='input')

    hidden_layer = model_input
    for i, layer_size in enumerate((128, 128)):
        hidden_layer = Dense(layer_size,
                             activation='relu',
                             name=f'hidden_{i}')(hidden_layer)

    value_fc = Dense(128, activation='relu', name='value_fc')(hidden_layer)
    value = Dense(1, name='value')(value_fc)

    advantage_fc = Dense(128, activation='relu', name='advantage_fc')(hidden_layer)
    advantage = Dense(MOVE_SPACE_SIZE, name='advantage')(advantage_fc)

    def aggregate(x):
        val, adv = x
        return val + adv - K.mean(adv, keepdims=True)

    model_output = Lambda(aggregate,
                          output_shape=(MOVE_SPACE_SIZE,))([value, advantage])

    model = Model(input=model_input, output=model_output)
    model.compile(optimizer=RMSprop(lr=learning_rate),
                  loss=tf.losses.huber_loss)
    # plot_model(model, to_file='model.png')
    # print(model.summary())

    return model


def sort_predictions(predictions):
    return sorted(zip(predictions, range(MOVE_SPACE_SIZE)), reverse=True)
