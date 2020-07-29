from typing import List, Tuple
from enigma.machine import EnigmaMachine
from faker import Faker
import re
from model import DeCryptModel
from decrypt_utils import *
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


def prepare_dataset(input_texts, target_texts, Tx=42):
    vocab, inv_vocab = get_vocabulary(input_texts, target_texts)
    X_one_hot, Y_one_hot = preprocess_data(input_texts, target_texts, vocab, Tx)
    return X_one_hot, Y_one_hot, vocab, inv_vocab


if __name__ == "__main__":
    import time

    # Generate plain and cipher data
    print('Data generation started .... ')
    plain, cipher = generate_data(1 << 14)
    print('Generated %s examples' % len(plain))

    X_train, X_test, y_train, y_test = train_test_split(cipher, plain, test_size=0.04, random_state=42)
    m = len(X_train)
    Tx = 42
    X_one_hot, Y_one_hot, vocab, inv_vocab = prepare_dataset(X_train, y_train, Tx)
    model = DeCryptModel(Xoh=X_one_hot, Yoh=Y_one_hot, Tx=Tx, m=m)

    inpt = input("- Press 1 To build model from scratch \n- Press 2 To load a pre trained model\n")
    if int(inpt) == 1:
        print('Training deep sequence model (ETA 10m) ....')
        start_time = time.time()
        # Train the model
        model.train(epochs=30)
        end_time = time.time()
        # Save the model
        model.save_model()
    else:
        start_time = time.time()
        print('Training deep sequence model (ETA 1minute) ....')
        model.load_model()
        end_time = time.time()

    print("************* S C O R E *************************")
    print('Decrypting holdout (test) samples ....')
    predicts_test = model.predict(X_test, vocab, inv_vocab)
    print("--- Test Sample score %s ---" % score(predicts_test, y_test))

    print('Decrypting all samples ....')
    predicts = model.predict(cipher, vocab, inv_vocab)
    print("--- Entire dataset score %s ---" % score(predicts, plain))
    print("************* S C O R E *************************")

    # log time taken to build model
    hours, rest = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rest, 60)
    print("--- Time taken for building the model: %s minutes  ---" % minutes)
