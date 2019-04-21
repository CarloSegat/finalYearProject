import random

import numpy as np
import sys

from TextPreprocessor import TextPreprocessor

sys.path.append('..\..\.')
from keras.models import Model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate, GRU
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.optimizers import Nadam

from ACDData import ACDData

from embeddings.Embeddings import Komn, Yelp, Google
from prepro import readfile, addCharInformation, padding, createBatches, createMatrices_syntax, iterate_minibatches_syntax
from utils import dump_gzip
from validation import compute_f1

use_syntax = False

EPOCHS = 90              # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 275     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended

all_scores = {'komn':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                     'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
             'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                        'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
            },

             'google':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    },
            'yelp':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    }
            }

class CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER
        self.text_preprocessor = TextPreprocessor()

    def loadData(self):
        """Load data and add character information"""
        self.trainSentences = readfile("data/NER-ABSA-16_Restaurants_Train.txt")
        #self.devSentences = readfile("data/dev.txt")
        self.testSentences = readfile("data/NER-ABSA-16_Restaurants_Test.txt")

    def addCharInfo(self):
        # format: [['EU', ['E', 'U'], 'B-ORG\n'], ...]
        self.trainSentences = addCharInformation(self.trainSentences)
        #self.devSentences = addCharInformation(self.devSentences)
        self.testSentences = addCharInformation(self.testSentences)

    def embed(self, syntax_x, syntax_test_x, embeddings, no_stop, no_punct):
        """Create word- and character-level embeddings"""

        # can call s.make_syntactical_vocabulary() to get unique syntactic_words
        labelSet, words = self.get_unique_labels_and_words()
        self.map_labels_to_indexes(labelSet)

        # mapping for token cases
        case2Idx = {'numeric': 0, 'allLower': 1, 'allUpper': 2, 'initialUpper': 3, 'other': 4, 'mainly_numeric': 5,
                    'contains_digit': 6, 'PADDING_TOKEN': 7}
        self.caseEmbeddings = np.identity(len(case2Idx), dtype='float32')  # identity matrix used

        # read GLoVE word embeddings
        word2Idx = {}
        self.wordEmbeddings = []

        # loop through each word in embeddings
        for word, vector in embeddings.word_to_emb.items():

            if len(word2Idx) == 0:  # add padding+unknown
                word2Idx["PADDING_TOKEN"] = len(word2Idx)
                vector = np.zeros(len(vector))  # zero vector for 'PADDING' word
                self.wordEmbeddings.append(vector)

                word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
                vector = np.random.uniform(-0.25, 0.25, len(vector))
                self.wordEmbeddings.append(vector)

            if word.lower() in words:
                vector = np.array(vector)
                self.wordEmbeddings.append(vector)  # word embedding vector
                word2Idx[word] = len(word2Idx)  # corresponding word dict

        self.wordEmbeddings = np.array(self.wordEmbeddings)

        # dictionary of all possible characters
        self.char2Idx = {"PADDING": 0, "UNKNOWN": 1}
        for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>â€“™Ã©˜¦":
            self.char2Idx[c] = len(self.char2Idx)

        self.train_set = padding(createMatrices_syntax(self.trainSentences,
                                 syntax_x, word2Idx, self.label2Idx,
                                 case2Idx, self.char2Idx, no_stop, no_punct,
                                 self.text_preprocessor))
        # self.dev_set = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        self.test_set = padding(createMatrices_syntax(self.testSentences, syntax_test_x,
                                word2Idx, self.label2Idx, case2Idx,
                                self.char2Idx, no_stop, no_punct,
                                self.text_preprocessor))

        # format: [[wordindices], [caseindices], [padded word indices], [label indices]]
        #  self.train_set = padding(createMatrices(self.trainSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        #  self.test_set = padding(createMatrices(self.testSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))

        self.idx2Label = {v: k for k, v in self.label2Idx.items()}

    def map_labels_to_indexes(self, labelSet):
        self.label2Idx = {}
        for label in labelSet:
            self.label2Idx[label] = len(self.label2Idx)

    def get_unique_labels_and_words(self):
        labelSet = set()
        words = {}
        # unique words and labels in data
        for dataset in [self.trainSentences,  self.testSentences]: #self.devSentences,
            for sentence in dataset:
                for token, char, label in sentence:
                    # token ... token, char ... list of chars, label ... BIO labels
                    labelSet.add(label)
                    words[token.lower()] = True
        return labelSet, words

    def createBatches(self):
        """Create batches"""
        self.train_batch, self.train_batch_len = createBatches(self.train_set)
       #self.dev_batch, self.dev_batch_len = createBatches(self.dev_set)
        self.test_batch, self.test_batch_len = createBatches(self.test_set)

    def tag_dataset(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            pred = model.predict([tokens, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def tag_dataset_syntax(self, dataset, model):
        """Tag data with numerical values"""
        correctLabels = []
        predLabels = []
        for i, data in enumerate(dataset):
            tokens, casing, char, labels, syntax = data
            tokens = np.asarray([tokens])
            casing = np.asarray([casing])
            char = np.asarray([char])
            syntax = np.asarray([syntax])
            if use_syntax:
                pred = model.predict([tokens, casing, char, syntax[:, :, 300:600]], verbose=False)[0]
            else:
                pred = model.predict([tokens, casing, char], verbose=False)[0]

            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def buildModel(self):
        """Model layers"""
        self.model = None
        # character input
        character_input = Input(shape=(None, 52,), name="Character_input")
        embed_char_out = TimeDistributed(
            Embedding(len(self.char2Idx), 30, embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)),
            name="Character_embedding")(
            character_input)

        dropout = Dropout(self.dropout)(embed_char_out)

        # CNN, time distributed allows us to map each "time stamp", i.e. character in a sentence, to an embedding.
        # Without time distributed the mapping would be between
        conv1d_out = TimeDistributed(
            Conv1D(kernel_size=self.conv_size, filters=30, padding='same', activation='tanh', strides=1),
            name="Convolution")(dropout)
        maxpool_out = TimeDistributed(MaxPooling1D(52), name="Maxpool")(conv1d_out)
        char = TimeDistributed(Flatten(), name="Flatten")(maxpool_out)
        char = Dropout(self.dropout)(char)

        # word-level input
        #words = Input(shape=(None, 600,), dtype='float32', name='words_input')
        words_input = Input(shape=(None,), dtype='int32', name='words_input')
        words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
                           weights=[self.wordEmbeddings],
                           trainable=False)(words_input)
        if use_syntax:
            syntax_input = Input(shape=(None, 300,))
            words = concatenate([words, syntax_input])
        # case-info input
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                           weights=[self.caseEmbeddings],
                           trainable=False)(casing_input)

        # concat & BLSTM
        output = concatenate([words, casing, char])
        output = Bidirectional(GRU(self.lstm_state_size,
                                   return_sequences=True,
                                   dropout=self.dropout,  # on input to each LSTM block
                                   recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                                   ), name="BGRU")(output)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        inputs = [words_input, casing_input, character_input]
        if use_syntax:
            inputs.append(syntax_input)
        self.model = Model(inputs=inputs, outputs=[output])
        #self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])
        self.optimizer = 'nadam'
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        #self.init_weights = self.model.get_weights()

        #plot_model(self.model, to_file='model.png')

        #print("Model built. Saved model.png\n")

    def train(self, use_syntax, max_epochs):
        """Default training"""

        self.f1_test_history = []

        for epoch in range(max_epochs):
            print("Epoch {}/{}".format(epoch, self.epochs))
            for i, batch in enumerate(iterate_minibatches_syntax(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char, syntax = batch
                if use_syntax:
                    self.model.train_on_batch([tokens, casing, char, syntax[:, :, 300:600]], labels)
                else:
                    self.model.train_on_batch([tokens, casing, char], labels)

            # compute F1 scores
            predLabels, correctLabels = self.tag_dataset_syntax(self.test_batch, self.model)
            pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, self.idx2Label)
            self.f1_test_history.append(f1_test)


            # predLabels, correctLabels = self.tag_dataset(self.dev_batch, self.model)
            # pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
            # self.f1_dev_history.append(f1_dev)
            # print("f1 dev ", round(f1_dev, 4), "\n")

        return max(self.f1_test_history)



cnn_blstm = CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
cnn_blstm.loadData()
cnn_blstm.addCharInfo()
# cnn_blstm.embed()
# cnn_blstm.createBatches()
# cnn_blstm.buildModel()
# cnn_blstm.train()

data = ACDData()
yelp = Yelp(data.make_normal_vocabulary())
komn = Komn(data.make_normal_vocabulary(), data.make_syntactical_vocabulary())
google = Google(data.make_normal_vocabulary())

syntax_train, y_train_val, syntax_test, y_test = data.get_data_syntax_concatenation(komn)
syntax_train, syntax_test = syntax_train[:, :, 300:600], syntax_test[:, :, 300:600]

batch_sizes = [32, 64, 96]

for p, punct in [(True, 'punct-removed'), (False, 'punct-kept')]:
    print(punct)
    for s, stop in [(True, 'stop-removed'), (False, 'stop-kept')]:
        print(stop)
        for embedding, emb in [(yelp, 'yelp'), (komn, 'komn'), (google, 'google')]:
            print(emb)
            cnn_blstm.embed(syntax_train, syntax_test, embedding, s, p)
            cnn_blstm.createBatches()
            for syntax, synt in [(True, 'synt'), (False, 'no-synt')]:
                print(synt)
                f1s = []
                for i in range(10):
                    batch_size = batch_sizes[random.randint(0, 2)]
                    cnn_blstm.buildModel()
                    f1s.append(cnn_blstm.train(synt))

                all_scores[emb][synt][stop][punct] = f1s

dump_gzip(all_scores, 'LSTM-ACD-Results-Yelp')

# best: 0.6008 f1