from typing import List, Tuple
from enigma.machine import EnigmaMachine
from faker import Faker
import re
from model import DeCryptModel
from nmt_utils import *
from sklearn.model_selection import train_test_split


class ConfiguredMachine:
    def __init__(self):
        self.machine = EnigmaMachine.from_key_sheet(
            rotors='II IV V',
            reflector='B',
            ring_settings=[1, 20, 11],
            plugboard_settings='AV BS CG DL FU HZ IN KM OW RX')

    def reset(self):
        self.machine.set_display('WXC')

    def encode(self, plain_str: str) -> str:
        self.reset()
        return self.machine.process_text(plain_str)

    def batch_encode(self, plain_list: List[str]) -> List[str]:
        encoded = list()
        for s in plain_list:
            encoded.append(self.encode(s))
        return encoded


def pre_process(input_str):
    return re.sub('[^a-zA-Z]', '', input_str).upper()


def generate_data(batch_size: int, seq_len: int = 42) -> Tuple[List[str], List[str]]:
    fake = Faker()
    machine = ConfiguredMachine()

    plain_list = fake.texts(nb_texts=batch_size, max_nb_chars=seq_len)
    plain_list = [pre_process(p) for p in plain_list]
    cipher_list = machine.batch_encode(plain_list)
    return plain_list, cipher_list


def predict(cipher_list: List[str]) -> List[str]:
    # solution here
    return cipher_list


def str_score(str_a: str, str_b: str) -> float:
    if len(str_a) != len(str_b):
        return 0

    n_correct = 0

    for a, b in zip(str_a, str_b):
        n_correct += int(a == b)

    return n_correct / len(str_a)


def score(predicted_plain: List[str], correct_plain: List[str]) -> float:
    correct = 0

    for p, c in zip(predicted_plain, correct_plain):
        if str_score(p, c) > 0.8:
            correct += 1

    return correct / len(correct_plain)


def prepare_dataset(input_texts, target_texts, m, Tx=42, Ty=42):
    dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(input_texts, target_texts, m)
    X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
    return X, Y, Xoh, Yoh, human_vocab, inv_machine_vocab


def prepare_test_dataset():
    test_cipher = []
    test_plain = []
    with open('test.txt', 'r', encoding='utf-8') as f:
        test_lines = f.read().split('\n')
    for line in test_lines:
        #     print(line.split('\t'))
        input_text, target_text = line.split('\t')
        test_cipher.append(input_text)
        test_plain.append(target_text)
    return test_cipher, test_plain


if __name__ == "__main__":
    plain, cipher = generate_data(1 << 14)

    X_train, X_test, y_train, y_test = train_test_split(cipher, plain, test_size=0.04, random_state=42)
    m = len(X_train)
    Tx = 42
    Ty = 42
    X, Y, Xoh, Yoh, human_vocab, inv_machine_vocab = prepare_dataset(X_train, y_train, m, Tx, Ty)

    model = DeCryptModel(Xoh=Xoh, Yoh=Yoh, Tx=Tx, Ty=Ty, m=m)
    model.train(epochs=10)

    predicts = model.predict(cipher, plain, human_vocab, inv_machine_vocab)
    train_score = score(predicts, plain)
    print(train_score)

    predicts_test = model.predict(X_test, y_test, human_vocab, inv_machine_vocab)
    test_score = score(predicts_test, y_test)

    print(test_score)


