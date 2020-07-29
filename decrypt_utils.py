import numpy as np
from keras.utils import to_categorical
from tqdm import tqdm
from constants import *


def get_vocabulary(input_texts, target_texts):
    """
        constructs vocabulary dictionary and inverse vocabulary dictionary from input sequences and output sequences
    """
    vocab = set()
    for i in tqdm(range(len(input_texts))):
        ip, tar = input_texts[i], target_texts[i]
        vocab.update(tuple(ip))
        vocab.update(tuple(tar))

    human = dict(zip(sorted(vocab) + [UNK, PAD], list(range(len(vocab) + 2))))
    inv_vocab = dict(enumerate(sorted(vocab) + [UNK, PAD]))
    return human, inv_vocab


def sequence_to_number(text, max_length, vocab):
    """
    transforms given string into a list of intergers based on vocab given
    eg: HELLO would be converted to [7, 4, 11, 11, 14, 27, 27 ...] 27 is <pad> representation

    Arguments:
    string -- input string, e.g. 'SCALEAI'
    length - max length of sequence
    vocab - dictionary representing character to integer representation

    Returns:
    number_representation -- list of integers representing the position of the string's character in the vocabulary
    """

    if len(text) > max_length:
        text = text[:max_length]

    number_representation = list(map(lambda x: vocab.get(x, UNK), text))

    if len(text) < max_length:
        number_representation += [vocab[PAD]] * (max_length - len(text))

    return number_representation


def preprocess_data(input_texts, target_texts, human_vocab, Tx):
    X = np.array([sequence_to_number(i, Tx, human_vocab) for i in input_texts])
    Y = [sequence_to_number(t, Tx, human_vocab) for t in target_texts]

    X_one_hot = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Y_one_hot = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), Y)))

    return X_one_hot, Y_one_hot


def preprocess_input(sequence, human_vocab, Tx):
    source = sequence_to_number(sequence, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = source.reshape((1, source.shape[0], source.shape[1]))
    return source
