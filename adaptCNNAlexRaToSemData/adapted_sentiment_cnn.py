import pdb
import numpy as np
import data_helpers
from w2v import train_word2vec
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D, \
    LSTM
from keras.layers.merge import Concatenate
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.preprocessing import sequence
import os, sys, inspect
import gzip
import pickle as pkl
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import loadData, classification_report

np.random.seed(0)

# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "keras_data_set"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 4, 5, 6)  # (3, 8)
num_filters = 100
dropout_prob = (0.5, 0.8)
hidden_dims = 50

batch_size = 64

sequence_length = 90

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

files = {
    "train": [("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016train-A.txt", 3),
              ("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016dev-A.txt", 3),
              ("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016devtest-A.txt", 3)],
    "test": [("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016test-A.txt", 3)]
}

x_train, y_train, x_test, y_test, vocabulary_inv, embeddings = loadData(files, 30000, cleaning=True)
file = gzip.open("../data/suresh_loaded_all.pkl.gz", "r")
data = pkl.load(file)
file.close()
all_w = data['all_word_to_emb']
assert(len(embeddings) == len(vocabulary_inv))
#if model_type in ["CNN-non-static", "CNN-static"]:
    #embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
    #                                   min_word_count=min_word_count, context=context)
    #if model_type == "CNN-static":
    #    x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
    #    x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
    #    print("x_train static shape:", x_train.shape)
    #    print("x_test static shape:", x_test.shape)
if model_type == "CNN-rand":
    embedding_weights = None


# Build model
if model_type == "CNN-static":
    input_shape = (sequence_length, embedding_dim)
else:
    input_shape = (sequence_length,)

model_input = Input(shape=input_shape)

# Static model does not have embedding layer
if model_type == "CNN-static":
    z = model_input
else:
    z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)
z = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(z)
z = Dropout(dropout_prob[0])(z)

# Convolutional block
conv_blocks = []
for sz in filter_sizes:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = MaxPooling1D(pool_size=2)(conv)
    # conv = GlobalMaxPooling1D()(conv) Global max pooling seem to be worse than MaxPooling
    conv = Flatten()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks)

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(3, activation="softmax")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
# Initialize weights with word2vec
if model_type == "CNN-non-static":
    #weights = np.array([v for v in embedding_weights.values()])
    #print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([embeddings])

# Train the model
y_train = to_categorical(y_train, num_classes=3, dtype='int32')
model.fit(np.array(x_train), y_train, batch_size=batch_size, epochs=4)
y_test_categorcal = to_categorical(y_test, num_classes=3, dtype='int32')
score, accuracy = model.evaluate(x_test, y_test_categorcal, batch_size=batch_size)
print("score: " + str(score))
print("acuracy: " + str(accuracy))
predicted = model.predict(x_test).argmax(axis=-1)

classification_report(y_test, predicted, [0, 1, 2])
