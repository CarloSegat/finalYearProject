from math import ceil
from random import shuffle
import numpy as np
from keras import callbacks
from keras.preprocessing.text import Tokenizer
import pickle as pkl
import gzip
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_tokenizer(sentences, wordsUsed):
    tokenizer = Tokenizer(wordsUsed)
    tokenizer.fit_on_texts(list(sentences))
    return tokenizer


def dump_gzip(data, output_file: str):
    f = gzip.open(output_file, 'wb')
    pkl.dump(data, f)
    f.close()

def load_gzip(file):
    file = gzip.open(file, "r")
    data = pkl.load(file)
    file.close()
    return data

def check_argument_is_numpy(arg):
    try:
        assert (type(arg).__module__ == np.__name__)
    except Exception:
        raise ValueError("Argument has to be numpy array")

def pad_array(arr, how_much):
    '''Pads with zeros'''
    p = np.zeros(len(arr[0]))
    for i in range(how_much - len(arr)):
        arr.append(p)
    assert(len(arr) == how_much)
    return arr

def split_train_x_y_and_validation(validation_size, x_train, y_train, classes=12):

    def get_classes_of_label(multi_label_vector):
        return np.where(multi_label_vector == 1)[0]

    how_many_of_each_class = ceil(len(x_train) * validation_size / classes)
    classes = dict(enumerate([0] * classes))
    d = list(zip(x_train, y_train))
    shuffle(d)
    train_x = []
    train_y = []
    validation_x = []
    validation_y = []

    for pair in d:
        put_in_validation = True
        for klass in get_classes_of_label(pair[1]):
            if classes[klass] > how_many_of_each_class:
                train_x.append(pair[0])
                train_y.append(pair[1])
                put_in_validation = False
                break
            else:
                continue
        if put_in_validation:
            for klass in get_classes_of_label(pair[1]):
                classes[klass] += 1
            validation_x.append(pair[0])
            validation_y.append(pair[1])

    return np.array(train_x), np.array(train_y), \
           (np.array(validation_x), np.array(validation_y))

def get_early_stop_callback():
    return callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=0, mode='auto',
                              restore_best_weights=True)





