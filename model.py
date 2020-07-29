from __future__ import print_function

import os

from keras import Input
from decrypt_utils import *
from keras.layers import Lambda, Dense, LSTM, Reshape
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from constants import *

# number of hidden state in LSTM
n_a = 128
# number of unique values in the vocabulary (26 + pading + unk) = 28
n_values = 28
reshapor = Reshape((1, n_values))
LSTM_cell = LSTM(n_a, return_state=True)
densor = Dense(n_values, activation='softmax')


def pluck_th_vector(x, t):
    return x[:, t, :]

class DeCryptModel:
    """
    Xoh - tensor with shape (m, Tx, vocab_size)
    Yoh - tensor with shape (m, Tx, vocab_size)
    Tx  - Max length of input sequence
    m - Number of samples
    """

    def __init__(self, Xoh, Yoh, Tx, m):
        self.Xoh = Xoh
        self.Yoh = Yoh
        self.Tx = Tx
        self.m = m
        self.model = None

    def __build_model__(self):
        """
        Model idea:
         - input layer: (m, Tx, n_values)
         - Generate each step of output (y<t>) using previous hidden states and inputs (x<t-1>)
         - output layer: (Ty, m, n_values)
        """

        X = Input(shape=(self.Tx, n_values))
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0
        # store the LSTM output after each of the Tx iterations into outputs
        outputs = []
        # Loop Tx times and on each step
        # - pluck `t`th time step vector from X using Lambda
        # pass the output to reshaper
        # perform LSTM step with n_a hidden states
        # send the result of LSTM to  Densly connected NN layer with softmax activation
        # persists all the outputs

        for t in range(self.Tx):
            x = Lambda(pluck_th_vector, output_shape=(28,), arguments={'t': t})(X)
            x = reshapor(x)
            a, _, c = LSTM_cell(x, initial_state=[a, c])
            out = densor(a)
            outputs.append(out)

        model = Model([X, a0, c0], outputs)
        return model

    def load_model(self, path=model_save_path):
        self.model = load_model(path)

    def train(self, epochs):
        self.model = self.__build_model__()
        # optimize using Adam
        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        s0, c0 = self._initial_states()
        outputs = list(self.Yoh.swapaxes(0, 1))
        self.model.fit([self.Xoh, s0, c0], outputs, epochs=epochs, batch_size=100)

    def save_model(self, path=model_save_path):
        self.model.save(path, overwrite=True)

    def predict(self, test_cipher, human_vocab, inv_machine_vocab):
        predicts = []
        for i in range(len(test_cipher)):
            source = preprocess_input(test_cipher[i], human_vocab, self.Tx)
            s0, c0 = self._initial_states()
            prediction = self.model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis=-1)
            output = []
            for k in prediction:
                if k == PAD_MAPPING:
                    break
                output.append(inv_machine_vocab[int(k)])
            predicts.append(''.join(output))
        return predicts

    def _initial_states(self):
        return np.zeros((self.m, n_a)), np.zeros((self.m, n_a))
