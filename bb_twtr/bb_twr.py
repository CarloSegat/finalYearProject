import collections
import gzip
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, GlobalMaxPooling1D, \
    LSTM
from keras.layers.merge import Concatenate
import os, sys, inspect
from keras import optimizers, callbacks
from bb_twtr.load_distant_supervision_amazon_data import AmazonSequence
from loss import f1_score
import loss

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import save_embeddings

np.random.seed(0)


# Model Hyperparameters
embedding_dim = 300
filter_sizes = [(1,2,3), (3,4,5), (5,6,7)]
num_filters = 200
dropout_prob = 0.5
hidden_dims = 30
batch_size = 64
sequence_length = 80

generator_train = AmazonSequence("../data/train_amazon")
generator_validation = AmazonSequence("../data/validation_amazon")

def make_model(train_flag = True):
    input_shape = (sequence_length,)
    model_input = Input(shape=input_shape)
    # when tuning the embeddign we want all the embeddings int he embeddings layer and not just a selection
    z = Embedding(number_entries_komn, embedding_dim, input_length=sequence_length,
                  name="embedding", trainable=train_flag)(model_input)

    # if fix_embeddings:
    #     z.trainable = False
    conv_blocks = []
    filter_sizes_to_use = filter_sizes[1]
    for sz in filter_sizes_to_use:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1,
                             kernel_initializer='orthogonal')(z)
        conv = GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    z = Concatenate()(conv_blocks)

    z = Dropout(dropout_prob)(z)
    z = Dense(hidden_dims, activation='sigmoid')(z) #TODO the paper doesnt say what activation function this layer has
    model_output = Dense(2, activation="softmax")(z)

    model = Model(model_input, model_output)

    embedding_layer = model.get_layer("embedding")
    embedding_layer.set_weights([embeddings])
    nadam = optimizers.nadam(clipnorm=1.)

    c = collections.Counter(y_train)
    weights = np.array([len(y_train)/c.get(0), len(y_train)/c.get(1), len(y_train)/c.get(2)])
    weighted_categorical_crossentropy = loss.weighted_categorical_crossentropy(weights)

    model.compile(loss=weighted_categorical_crossentropy, optimizer=nadam, metrics=[f1_score])
    print(model.summary())
    return model

model = make_model(False)

es = callbacks.EarlyStopping(monitor='val_f1_score',
                              patience=2,
                              verbose=1)

parameters_path = "first_epoch.hdf5"
checkpoint = ModelCheckpoint(parameters_path, monitor='val_f1_score', verbose=1, save_best_only=True, mode='max')

model.fit_generator(generator=generator_train,
                                          steps_per_epoch=(66137160 // batch_size),
                                          epochs=1,
                                          verbose=1,
                                          validation_data=generator_validation,
                                          validation_steps=(16539979 // batch_size),
                                          use_multiprocessing=True,
                                          workers=16,
                                          max_queue_size=32,
                                          callbacks=[checkpoint])

model = make_model()
model.load_weights(parameters_path)
model.fit_generator(generator=generator_train,
                                          steps_per_epoch=(66137160 // batch_size),
                                          epochs=6,
                                          verbose=1,
                                          validation_data=generator_validation,
                                          validation_steps=(16539979 // batch_size),
                                          use_multiprocessing=True,
                                          workers=16,
                                          max_queue_size=32,)

save_embeddings(model)

