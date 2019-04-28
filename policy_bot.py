import os
from typing import Tuple, Optional

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam

from utils import (Mark, Action, Result, ALL_MOVES,
                   get_encoded_invalid_moves,
                   STATE_SPACE_SIZE, MOVE_SPACE_SIZE, encode_board)


class Player:
    def __init__(self, mark: Mark, *,
                 gamma: float = 0.99,
                 training: bool = False,
                 learning_rate: float = 0.001,
                 weights_file: Optional[str] = './weights/policy.h5') -> None:
        self.mark = mark
        self.gamma = gamma
        self.training = training

        self.model = build_model(learning_rate)
        if weights_file and os.path.isfile(weights_file):
            self.load(weights_file)

        self.boards = np.empty(0).reshape(0, STATE_SPACE_SIZE)
        self.moves = np.empty(0).reshape(0, 1)

    def move(self, board) -> Tuple[int, int, Action]:
        board_hash = encode_board(board, self.mark)

        predictions = self.model.predict(board_hash)[0]
        predictions[get_encoded_invalid_moves(board, self.mark)] = 0.0
        norm = np.linalg.norm(predictions, ord=1)
        if norm == 0:
            predictions[0] = 1.0
        else:
            predictions /= norm

        if self.training:
            move_hash = np.random.choice(range(MOVE_SPACE_SIZE), p=predictions)
            self.boards = np.vstack([self.boards, board_hash])
            self.moves = np.vstack([self.moves, move_hash])
            return ALL_MOVES[move_hash]
        else:
            return ALL_MOVES[np.argmax(predictions)]

    def train(self, result: Result) -> None:
        moves = self.moves.flatten().astype(int)
        moves_one_hot = np.zeros((len(moves), MOVE_SPACE_SIZE))
        moves_one_hot[np.arange(len(moves)), moves] = 1

        rewards = np.array([result.value * self.gamma**i
                            for i in reversed(range(len(moves)))])
        self.model.train([self.boards, moves_one_hot, rewards])

        self.boards = np.empty(0).reshape(0, STATE_SPACE_SIZE)
        self.moves = np.empty(0).reshape(0, 1)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def build_model(learning_rate: float = 0.001) -> Model:
    input_layer = Input(shape=(STATE_SPACE_SIZE,), name='input_layer')

    hidden_layer = input_layer
    for i, neurons_num in enumerate((128, 128)):
        hidden_layer = Dense(neurons_num,
                             activation='relu',
                             use_bias=False,
                             name=f'hidden_layer{i}')(hidden_layer)

    output_layer = Dense(MOVE_SPACE_SIZE,
                         activation='softmax',
                         use_bias=False,
                         name='output_layer')(hidden_layer)

    model = Model(inputs=[input_layer], outputs=output_layer)

    move_one_hot = K.placeholder(shape=(None, MOVE_SPACE_SIZE),
                                 name='move_one_hot')
    reward = K.placeholder(shape=(None,), name='reward')

    log_move_probability = K.log(K.sum(output_layer * move_one_hot, axis=1))

    optimizer = Adam(lr=learning_rate)
    loss = K.mean(-log_move_probability * reward)
    updates = optimizer.get_updates(loss=loss, params=model.trainable_weights)

    model.train = K.function(inputs=[input_layer, move_one_hot, reward],
                             outputs=[],
                             updates=updates)

    return model
