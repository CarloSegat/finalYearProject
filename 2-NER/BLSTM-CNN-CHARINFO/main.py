import numpy as np
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Conv1D, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, MaxPooling1D, \
    Flatten, concatenate
from gensim.models import word2vec, KeyedVectors
from keras.utils import plot_model
from keras.initializers import RandomUniform
from keras.optimizers import SGD, Nadam

from SemEval import SemEvalData
from embeddings.Embeddings import Komn
from prepro import readfile, addCharInformation, padding, createMatrices, createBatches, iterate_minibatches, \
    createMatrices_syntax, iterate_minibatches_syntax
from utils import load_gzip, dump_gzip
from validation import compute_f1

EPOCHS = 90              # paper: 80
DROPOUT = 0.5             # paper: 0.68
DROPOUT_RECURRENT = 0.25  # not specified in paper, 0.25 recommended
LSTM_STATE_SIZE = 275     # paper: 275
CONV_SIZE = 3             # paper: 3
LEARNING_RATE = 0.0105    # paper 0.0105
OPTIMIZER = Nadam()       # paper uses SGD(lr=self.learning_rate), Nadam() recommended

class CNN_BLSTM(object):

    def __init__(self, EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER):

        self.epochs = EPOCHS
        self.dropout = DROPOUT
        self.dropout_recurrent = DROPOUT_RECURRENT
        self.lstm_state_size = LSTM_STATE_SIZE
        self.conv_size = CONV_SIZE
        self.learning_rate = LEARNING_RATE
        self.optimizer = OPTIMIZER

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

    def embed(self):
        """Create word- and character-level embeddings"""

        s = SemEvalData()
        k = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
        syntax_x, _, syntax_test_x, _ = s.get_data_syntax_concatenation(k)
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
        for word, vector in k.word_to_emb.items():

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

        self.train_set = padding(createMatrices_syntax(self.trainSentences, syntax_x, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        # self.dev_set = padding(createMatrices(self.devSentences, word2Idx, self.label2Idx, case2Idx, self.char2Idx))
        self.test_set = padding(createMatrices_syntax(self.testSentences, syntax_test_x, word2Idx, self.label2Idx, case2Idx, self.char2Idx))

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
            pred = model.predict([syntax, casing, char], verbose=False)[0]
            pred = pred.argmax(axis=-1)  # Predict the classes
            correctLabels.append(labels)
            predLabels.append(pred)
        return predLabels, correctLabels

    def buildModel(self):
        """Model layers"""

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
        words = Input(shape=(None, 600,), dtype='float32', name='words_input')
        # words_input = Input(shape=(None,), dtype='int32', name='words_input')
        # words = Embedding(input_dim=self.wordEmbeddings.shape[0], output_dim=self.wordEmbeddings.shape[1],
        #                    weights=[self.wordEmbeddings],
        #                    trainable=False)(words_input)

        # case-info input
        casing_input = Input(shape=(None,), dtype='int32', name='casing_input')
        casing = Embedding(output_dim=self.caseEmbeddings.shape[1], input_dim=self.caseEmbeddings.shape[0],
                           weights=[self.caseEmbeddings],
                           trainable=False)(casing_input)

        # concat & BLSTM
        output = concatenate([words, casing, char])
        output = Bidirectional(LSTM(self.lstm_state_size,
                                    return_sequences=True,
                                    dropout=self.dropout,  # on input to each LSTM block
                                    recurrent_dropout=self.dropout_recurrent  # on recurrent input signal
                                    ), name="BLSTM")(output)
        output = TimeDistributed(Dense(len(self.label2Idx), activation='softmax'), name="Softmax_layer")(output)

        # set up model
        self.model = Model(inputs=[words, casing_input, character_input], outputs=[output])
        #self.model = Model(inputs=[words_input, casing_input, character_input], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        self.init_weights = self.model.get_weights()

        plot_model(self.model, to_file='model.png')

        print("Model built. Saved model.png\n")

    def train(self):
        """Default training"""

        self.f1_test_history = []
        #self.f1_dev_history = []

        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch, self.epochs))
            for i, batch in enumerate(iterate_minibatches_syntax(self.train_batch, self.train_batch_len)):
                labels, tokens, casing, char, syntax = batch
                self.model.train_on_batch([syntax, casing, char], labels)

            # compute F1 scores
            predLabels, correctLabels = self.tag_dataset_syntax(self.test_batch, self.model)
            pre_test, rec_test, f1_test = compute_f1(predLabels, correctLabels, self.idx2Label)
            self.f1_test_history.append(f1_test)
            print("f1 test ", round(f1_test, 4))

            # predLabels, correctLabels = self.tag_dataset(self.dev_batch, self.model)
            # pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, self.idx2Label)
            # self.f1_dev_history.append(f1_dev)
            # print("f1 dev ", round(f1_dev, 4), "\n")

        print("Final F1 test score: ", f1_test)

        print("Training finished.")

        # save model
        self.modelName = "{}_{}_{}_{}_{}_{}_{}".format(self.epochs,
                                                       self.dropout,
                                                       self.dropout_recurrent,
                                                       self.lstm_state_size,
                                                       self.conv_size,
                                                       self.learning_rate,
                                                       self.optimizer.__class__.__name__
                                                       )

        modelName = self.modelName + ".h5"
        self.model.save(modelName)
        print("Model weights saved.")

        self.model.set_weights(self.init_weights)  # clear model
        print("Model weights cleared.")
        print("best F1 score for test was: ", max(self.f1_test_history))

    def writeToFile(self):
        """Write output to file"""

        # .txt file format
        # [epoch  ]
        # [f1_test]
        # [f1_dev ]

        output = np.matrix([[int(i) for i in range(self.epochs)], self.f1_test_history]) #, self.f1_dev_history

        fileName = self.modelName + ".txt"
        with open(fileName, 'wb') as f:
            for line in output:
                np.savetxt(f, line, fmt='%.5f')

        print("Model performance written to file.")

    print("Class initialised.")

cnn_blstm = CNN_BLSTM(EPOCHS, DROPOUT, DROPOUT_RECURRENT, LSTM_STATE_SIZE, CONV_SIZE, LEARNING_RATE, OPTIMIZER)
cnn_blstm.loadData()
cnn_blstm.addCharInfo()
cnn_blstm.embed()
cnn_blstm.createBatches()
cnn_blstm.buildModel()
cnn_blstm.train()
cnn_blstm.writeToFile()

# best: 0.6008 f1