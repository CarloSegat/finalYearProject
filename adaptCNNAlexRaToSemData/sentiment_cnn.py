"""
Train convolutional network for sentiment analysis on IMDB corpus. Based on
"Convolutional Neural Networks for Sentence Classification" by Yoon Kim
http://arxiv.org/pdf/1408.5882v2.pdf

For "CNN-rand" and "CNN-non-static" gets to 88-90%, and "CNN-static" - 85% after 2-5 epochs with following settings:
embedding_dim = 50          
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

Differences from original article:
- larger IMDB corpus, longer sentences; sentence length is very important, just like data size
- smaller embedding dimension, 50 instead of 300
- 2 filter sizes instead of original 3
- fewer filters; original work uses 100, experiments show that 3-10 is enough;
- random initialization is no worse than word2vec init on IMDB corpus
- sliding Max Pooling instead of original Global Pooling
"""
import pdb
import numpy as np
import data_helpers
from w2v import train_word2vec
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D
from keras.layers.merge import Concatenate
from keras.datasets import imdb
from keras.preprocessing import sequence
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import loadData

np.random.seed(0)

# Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static

# Data source
data_source = "keras_data_set"  # keras_data_set|local_dir

# Model Hyperparameters
embedding_dim = 300
filter_sizes = (3, 4, 5, 6)#(3, 8)
num_filters = 100
dropout_prob = (0.5, 0.8)
hidden_dims = 50

batch_size = 64

sequence_length = 90

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10

pathA = "../data/SEData/2017/4a-english/4A-English/"
pathB = "../data/SEData/2017/4b-english/4B-English/"
pathCE = "../data/SEData/2017/4c-english/4C-English/" #"SemEval2017-task4-dev.subtask-BD.english.INPUT.txt"

path2016 = "../data/SEData/2016/devtest/"
files = {
	"train":(pathA + 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt', 3),
	"test" :(path2016 + '100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt', 3)
}

x_train, y_train, x_test, y_test, vocabulary_inv = loadData(files, 20000, cleaning=True)
pdb.set_trace()

if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")

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
	#conv = GlobalMaxPooling1D()(conv) Global max pooling seem to be worse than MaxPooling
	conv = Flatten()(conv)
	conv_blocks.append(conv)
z = Concatenate()(conv_blocks) 

z = Dropout(dropout_prob[1])(z)
z = Dense(hidden_dims, activation="relu")(z)
model_output = Dense(1, activation="sigmoid")(z)

model = Model(model_input, model_output)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Initialize weights with word2vec
if model_type == "CNN-non-static":
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([weights])

# Train the model
model.fit(np.array(x_train), np.array(y_train), batch_size=batch_size, epochs=4)
score, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
print("score: " + str(score))
print("acuracy: " + str(accuracy))