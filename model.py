from __future__ import print_function

from nmt_utils import *
from keras.layers import Lambda, Dense, LSTM, Reshape
from keras.utils import to_categorical
import numpy as np
from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam

n_a = 128
n_values = 28
n_s = 128
reshapor = Reshape((1, 28))  # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state=True)  # Used in Step 2.C
densor = Dense(n_values, activation='softmax')


class DeCryptModel:
    def __init__(self, Xoh, Yoh, Tx, Ty, m):
        self.Xoh = Xoh
        self.Yoh = Yoh
        self.Tx = Tx
        self.Ty = Ty
        self.m = m
        self.model = None

    def __model__(self, Tx, n_a, n_values):
        """
        Implement the model

        Arguments:
        Tx -- length of the sequence in a corpus
        n_a -- the number of activations used in our model
        n_values -- number of unique values in the music data

        Returns:
        model -- a keras model with the
        """

        # Define the input of your model with a shape
        X = Input(shape=(Tx, n_values))

        # Define s0, initial hidden state for the decoder LSTM
        a0 = Input(shape=(n_a,), name='a0')
        c0 = Input(shape=(n_a,), name='c0')
        a = a0
        c = c0

        ### START CODE HERE ###
        # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
        outputs = []

        # Step 2: Loop
        for t in range(Tx):
            # Step 2.A: select the "t"th time step vector from X.
            x = Lambda(lambda x: X[:, t, :])(X)
            # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
            x = reshapor(x)
            # Step 2.C: Perform one step of the LSTM_cell
            a, _, c = LSTM_cell(x, initial_state=[a, c])
            # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
            out = densor(a)
            # Step 2.E: add the output to "outputs"
            outputs.append(out)

        # Step 3: Create model instance
        model = Model([X, a0, c0], outputs)
        return model

    def load_model(self, path='Decrypt.h5'):
        self.model.load_weights(path)

    def train(self, epochs):
        self.model = self.__model__(Tx=42, n_a=128, n_values=28)
        opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        s0 = np.zeros((self.m, n_s))
        c0 = np.zeros((self.m, n_s))
        outputs = list(self.Yoh.swapaxes(0, 1))
        self.model.fit([self.Xoh, s0, c0], outputs, epochs=epochs, batch_size=100)

    def save_model(self):
        self.model.save('Decrypt.h5', overwrite=True)

    def predict(self, test_cipher, test_plain, human_vocab, inv_machine_vocab):
        predicts = []
        for i in range(len(test_cipher)):
            source = string_to_int(test_cipher[i], self.Tx, human_vocab)
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
            source = source.reshape((1, source.shape[0], source.shape[1]))

            s0 = np.zeros((self.m, n_s))
            c0 = np.zeros((self.m, n_s))

            prediction = self.model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis=-1)
            output = []
            for k in prediction:
                if k == 27:
                    break
                output.append(inv_machine_vocab[int(k)])
            predicts.append(''.join(output))
        return predicts
