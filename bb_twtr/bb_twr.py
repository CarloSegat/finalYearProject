import pdb
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D, \
    LSTM
from keras.layers.merge import Concatenate
from keras.utils import to_categorical
from keras.datasets import imdb
from keras.preprocessing import sequence
import os, sys, inspect, random
from keras import optimizers
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import loadData, classification_report

np.random.seed(0)


# Model Hyperparameters
embedding_dim = 300
filter_sizes = [(1,2,3), (3,4,5), (5,6,7)]
num_filters = 200
dropout_prob = 0.5
hidden_dims = 30
batch_size = 64
sequence_length = 80

# update embeddings?
fix_embeddings = True

files = {
    "train": [("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016train-A.txt", 3),
              ("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016dev-A.txt", 3),
              ("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016devtest-A.txt", 3)],
    "test": [("../data/SEData/2017/englishTrainingData/Subtask_A/twitter-2016test-A.txt", 3)]
}

x_train, y_train, x_test, y_test, vocabulary_inv, embeddings = loadData(files, 30000,  maxLen=sequence_length, cleaning=True)
assert(len(embeddings) == len(vocabulary_inv))

input_shape = (sequence_length,)
model_input = Input(shape=input_shape)
z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

if fix_embeddings:
    z.trainable = False
conv_blocks = []
filter_sizes_to_use = filter_sizes[random.randint(0,2)]
for sz in filter_sizes_to_use:
    conv = Convolution1D(filters=num_filters,
                         kernel_size=sz,
                         padding="valid",
                         activation="relu",
                         strides=1)(z)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)
z = Concatenate()(conv_blocks)

z = Dropout(dropout_prob)(z)
z = Dense(hidden_dims)(z) #TODO the paper doesnt say what activation function this layer has
model_output = Dense(3, activation="softmax")(z)

model = Model(model_input, model_output)

embedding_layer = model.get_layer("embedding")
embedding_layer.set_weights([embeddings])
nadam = optimizers.nadam(clipnorm=1.)
model.compile(loss="categorical_crossentropy", optimizer=nadam, metrics=["accuracy"])
print(model.summary())

y_train = to_categorical(y_train, num_classes=3, dtype='int32')
model.fit(np.array(x_train), y_train, batch_size=batch_size, epochs=20)
y_test_categorcal = to_categorical(y_test, num_classes=3, dtype='int32')
score, accuracy = model.evaluate(x_test, y_test_categorcal, batch_size=batch_size)
print("score: " + str(score))
print("acuracy: " + str(accuracy))
predicted = model.predict(x_test).argmax(axis=-1)

classification_report(y_test, predicted, [0, 1, 2])
