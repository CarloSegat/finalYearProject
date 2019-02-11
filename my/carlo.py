import pandas as pd
import re
import numpy as np
import pdb
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.models import Model
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers import Dense, GlobalMaxPooling1D, Activation, Dropout, GaussianNoise
from keras.layers import Embedding, Input, BatchNormalization, SpatialDropout1D, Conv1D
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pandas_summary import DataFrameSummary 
from IPython.display import display
import itertools
from nltk.corpus import words
import matplotlib.pyplot as plt
from utils import *
from gensim.models import *

### Set parameters
embed_size   = 300    # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen       = 100   # max number of words in a comment to use
myActivationFunction = 'tanh'	

### Load data
list_sentences_train = loadSentences('..\data\SEData\2017\4a-english\4A-English\SemEval2017-task4-dev.subtask-A.english.INPUT.txt')
list_sentences_test = loadSentences("'..\data\SEData\2016\devtest\100_topics_100_tweets.sentence-three-point.subtask-A.devtest.gold.txt")
y = loadTargets('..\data\SEData\2017\4a-english\4A-English\SemEval2017-task4-dev.subtask-A.english.INPUT.txt')	
assert(y.shape[0] == len(list_sentences_train))
list_classes = ["positive", "neutral", "negative"]

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
#list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post')
#X_test = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')

googleModel = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
wv = googleModel.wv

word_index = tokenizer.word_index # Word index stores the words we have tokenised, most common words have higher indexes
nb_words = min(max_features, len(word_index))

# Initialize embedding matrix with random arrays whose elemnts are drawn from a normal distribution with 
# parameters based on the predefined embedding values
embedding_matrix = np.random.normal(0, 3, (nb_words, embed_size)) 

# Loop through each word and get its embedding vector
for word, wordIndex in word_index.items(): # list of tuples
	if wordIndex >= max_features:
		continue # Skip words appearing less than the minimum allowed
	try:
		embedding_vector = wv[word]
		embedding_matrix[wordIndex] = embedding_vector
	except KeyError:
		embedding_matrix[wordIndex] = embedding_matrix[wordIndex]
		
## ????????????
conv_filters = 128 # No. filters to use for each convolution


sampleLength = X_train.shape[1] # 100 as specified in max length
inp = Input(shape=(sampleLength,), dtype='int64')
# weights represents a pre-trained word embeddings
emb = Embedding(input_dim=max_features, output_dim=embed_size, weights=[embedding_matrix])(inp)

# Specify each convolution layer and their kernel siz i.e. n-grams 
conv1_1 = Conv1D(filters=conv_filters, kernel_size=3)(emb)
# Output dimensions (batchSize, embed_size - kernel_size + 1, filters) 
btch1_1 = BatchNormalization()(conv1_1)
actv1_1 = Activation(myActivationFunction)(btch1_1) 
glmp1_1 = GlobalMaxPooling1D()(actv1_1)

conv1_2 = Conv1D(filters=conv_filters, kernel_size=4)(emb)
btch1_2 = BatchNormalization()(conv1_2)
actv1_2 = Activation('relu')(btch1_2)
glmp1_2 = GlobalMaxPooling1D()(actv1_2)

conv1_3 = Conv1D(filters=conv_filters, kernel_size=5)(emb)
btch1_3 = BatchNormalization()(conv1_3)
actv1_3 = Activation('relu')(btch1_3)
glmp1_3 = GlobalMaxPooling1D()(actv1_3)

conv1_4 = Conv1D(filters=conv_filters, kernel_size=6)(emb)
btch1_4 = BatchNormalization()(conv1_4)
actv1_4 = Activation('relu')(btch1_4)
glmp1_4 = GlobalMaxPooling1D()(actv1_4)

# Gather all convolution layers
cnct = concatenate([glmp1_1, glmp1_2, glmp1_3, glmp1_4], axis=1)
drp1 = Dropout(0.2)(cnct)
dns1  = Dense(32, activation='relu')(drp1)
btch1 = BatchNormalization()(dns1)
drp2  = Dropout(0.2)(btch1)
out = Dense(y.shape[1], activation='sigmoid')(drp2) # output array of 3 probabilities

# Compile
model = Model(inputs=inp, outputs=out)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#pdb.set_trace()

# Estimate model
model.fit(X_train, y, validation_split=0.2, epochs=10, batch_size=64, shuffle=True)
pdb.set_trace()
# Predict
preds = model.predict(X_test)

submid = pd.DataFrame({'id': test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = list_classes)], axis=1)
submission.to_csv('conv_glove_simple_sub.csv', index=False)