import pandas as pd
import re
import numpy as np
import pdb
from keras.preprocessing import sequence
# what is it?
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

### Read the glove word vectors (space delimited strings) into a dictionary from word->vector

### Set parameters
embed_size   = 50    # how big is each word vector
max_features = 20000 # how many unique words to use (i.e num rows in embedding vector)
maxlen       = 100   # max number of words in a comment to use
myActivationFunction = 'tanh'

### Load data
train = pd.read_csv('train.csv') 
test = pd.read_csv('test.csv') 
list_sentences_train = train["comment_text"].fillna("_NaN_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values # [[0,0,0,0,0][...]]
list_sentences_test = test["comment_text"].fillna("_NaN_").values
### Convert sentences to padded list of integers
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# The Tokenizer stores everything in the word_index during fit_on_texts. 
# Then, when calling the texts_to_sequences method, only the top num_words are considered.
# i.e. if list_sentences_train has more than num_word eords in excess wont be in the word_index and
# they will be ignored when calling texts_to_sequences
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_train = pad_sequences(list_tokenized_train, maxlen=maxlen, padding='post')
pdb.set_trace()
X_test = pad_sequences(list_tokenized_test, maxlen=maxlen, padding='post')

### Read a list of 40.000 embeddings, word --> vector of floats
embeddings_index = {}
for o in open('./glove.6B.50d.txt', encoding="utf8"):
	embeddings_index[o.strip().split()[0]] = np.asarray(o.strip().split()[1::], dtype='float32')


### Create embeddings matrix
all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index # Word index stores the words we have tokenised, most common words have higher indexes
nb_words = min(max_features, len(word_index))

# Initialize embedding matrix with random arrays whose elemnts are drawn from a normal distribution with 
# parameters based on the predefined embedding values
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)) 

# Loop through each word and get its embedding vector
for word, wordIndex in word_index.items(): # list of tuples
	if wordIndex >= max_features:
		continue # Skip words appearing less than the minimum allowed
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None: 
		embedding_matrix[wordIndex] = embedding_vector
		
## ????????????
conv_filters = 128 # No. filters to use for each convolution
weight_vec = list( np.max(np.sum(y, axis=0)) / np.sum(y, axis=0) ) # axis 0 is vertical
class_weight = {i: weight_vec[i] for i in range(6)}


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
out = Dense(y.shape[1], activation='sigmoid')(drp2)

# Compile
model = Model(inputs=inp, outputs=out)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
#pdb.set_trace()

# Estimate model
model.fit(X_train, y, validation_split=0.1, epochs=2, batch_size=32, shuffle=True, class_weight=class_weight)

# Predict
preds = model.predict(X_test)

submid = pd.DataFrame({'id': test["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = list_classes)], axis=1)
submission.to_csv('conv_glove_simple_sub.csv', index=False)