import gzip
import random

from keras.utils import Sequence
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from utils import ROOT_DIR, preprocess, load_gzip, map_vocabulary_to_embeddings, dump_gzip
import numpy as np
import pickle as pkl

class AmazonSequence(Sequence):
    '''
    Generates sequences of review texts which are 80 characters or less after tokenization
    The labels are 0 and 1 for negative and positive reviews respecetivelly.
    '''

    def __init__(self, file_name,  batch_size=32, max_sentence_length=80):
        self.data_iterator = open(ROOT_DIR + file_name)
        self.batch_size = batch_size
        self.max_sentence_length = max_sentence_length
        self.stopwords = stopwords.words('english')
        self.amazon_dictionary = open(ROOT_DIR + '/data/amazon_dictionary.pkl.gz', 'r')

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for review in self.data_iterator:
            clean, review = self.get_clean_review(review)
            #clean =
            if len(clean) > self.max_sentence_length:
                # Amazon review is too long
                continue
            else:
                y = self.get_label(review)
                x = self.apply_padding(clean)
                batch_x.append(x)
                batch_y.append(y)
                if len(batch_x) == self.batch_size:
                    return np.array([batch_x]), np.array(batch_y)

    def apply_padding(self, clean):
        padding_total = self.max_sentence_length - len(clean)
        pad_left = padding_total // 2
        pad_right = padding_total - pad_left
        x = pad_sequences(clean, pad_left, padding='pre')
        x = pad_sequences(x, pad_right, padding='post')
        assert(len(x) == self.max_sentence_length)
        return x

    def get_label(self, review):
        if review["overall"] <= 2.0:
            y = 0
        elif review["overall"] >= 4.0:
            y = 1
        return y

    def get_clean_review(self, review):
        review = eval(review)
        sentence = review["reviewText"]
        clean = preprocess(sentence)
        return clean, review


def split_amazon_data(percentage=0.2):
    file_name = "../data/user_dedup.json"
    f = open(file_name)
    t = open("train_amazon", "w+")
    v = open("validation_amazon", "w+")
    while True:
        try:
            element = next(f)
            element_eval = eval(element)
            if element_eval["overall"] == 5.0 or element_eval["overall"] == 1.0:
                if random.random() < percentage:
                    v.write(element)
                else:
                    t.write(element)
        except:
            t.close()
            v.close()
            break

def count():
    stoplist = stopwords.words('english')
    t = open("train_amazon", "r")
    train_positive = 0
    train_negative = 0
    validation_positive = 0
    validation_negative = 0
    wasted = 0
    while True:
        try:
            sentence = next(t)
            sentence = eval(sentence)
            review = sentence["reviewText"]
            clean = [word for word in review.split() if word not in stoplist]
            if len(clean) <= 80:
                if sentence["overall"] <= 2.0:
                    train_negative += 1
                elif sentence["overall"] >= 4.0:
                    train_positive += 1
                else:
                    wasted += 1
            else:
                wasted += 1
        except:
            t.close()
            break


    v = open("validation_amazon", "r")
    while True:
        try:
            sentence = next(t)
            clean = [word for word in sentence.split() if word not in stoplist]
            if len(clean) <= 80:
                if sentence["overall"] <= 2.0:
                    validation_negative += 1
                elif sentence["overall"] >= 4.0:
                    validation_positive += 1
                else:
                    wasted += 1
            else:
                wasted += 1
        except:
            v.close()
            break
    print("train_positive", train_positive,
            "train_negative", train_negative,
            "validation_positive", validation_positive,
            "validation_negative", validation_negative,
            "wasted", wasted)

def build_dictionary_amazon():
    output_file = 'amazon_dictionary.pkl.gz'
    t = "train_amazon"
    v = "validation_amazon"
    vocabulary = {}
    c = 0

    ptr, ntr = 0, 0
    openFile = open(t, 'r')
    for line in openFile:
        c += 1
        sentence = eval(line)
        review = sentence["reviewText"]
        if sentence["overall"] == 5.0:
            ptr += 1
        elif sentence["overall"] == 1.0:
            ntr += 1
        review = preprocess(review)
        review = review.split(" ")
        for w in review:
            try:
                vocabulary[w] = vocabulary[w] + 1
            except:
                vocabulary[w] = 1
    openFile.close()

    pte, nte = 0, 0
    openFile = open(v, 'r')
    for line in openFile:
        c += 1
        sentence = eval(line)
        review = sentence["reviewText"]
        if sentence["overall"] == 5.0:
            pte += 1
        elif sentence["overall"] == 1.0:
            nte += 1
        review = preprocess(review)
        review = review.split(" ")
        for w in review:
            try:
                vocabulary[w] = vocabulary[w] + 1
            except:
                vocabulary[w] = 1
    openFile.close()

    data = {'amazon_vocabulary': vocabulary,
            'number_positive_train': ptr,
            'number_positive_test': pte,
            'number_negative_train': ntr,
            'number_negative_test': nte}
    f = gzip.open(output_file, 'wb')
    pkl.dump(data, f)
    f.close()

def remove_singleton(amazon_dictionary = 'amazon_dictionary.pkl.gz'):
    output_file = 'amazon_vocabulary_no_sigleton.pkl.gz'
    dictionary = gzip.open(amazon_dictionary, 'r')
    dictionary = pkl.load(dictionary)
    #dictionary = eval(dictionary)
    dictionary = dictionary['amazon_vocabulary']
    words_to_remove = []
    for k, v in dictionary.items():
        if v == 1:
            words_to_remove.append(k)
    for word in words_to_remove:
        del dictionary[word]
    dump_gzip('amazon_vocabulary_no_sigleton', dictionary, output_file)

def create_word_to_int_for_amazon():
    '''
    V = vocabulary of amazon reviewws
    EMBS = komn embeddings
    This method maps V to different integers where EMBS[integer] = word embedding for the word mapped to int
    '''
    embs = load_gzip()
    file = gzip.open('amazon_vocabulary_no_sigleton.pkl.gz', "r")
    data = pkl.load(file)
    file.close()
    vocabulary = data['amazon_vocabulary_no_sigleton']
    words = vocabulary.keys()
    word_to_int = map_vocabulary_to_embeddings(embs, words)

build_dictionary_amazon()