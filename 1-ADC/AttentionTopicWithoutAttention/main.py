from keras import Input, Model, metrics
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, Bidirectional, GRU
from keras.optimizers import SGD
from keras import backend as K
from tensorflow import norm
import tensorflow as tf
import numpy as np
from embeddings.Embeddings import Yelp, Komn
from SemEval import SemEvalData


# Paper Parameters
from utils import split_train_x_y_and_validation, get_early_stop_callback

VALIDATION_PERCENTAGE = 0.1
TOPICS = 11
DROPOUT = 0.4
MAX_SEQUENCE_LENGTH = 80
GRU_HIDDEN_SIZE = 128
SIZE_OF_TOPIC_VECTOR = GRU_HIDDEN_SIZE * 2
NEURONS_MLP_SQUASH_1_2016 = 32
NEURONS_MLP_SQUASH_2_2016 = 64
NEURONS_MLP_SQUASH_1_2014 = 16
NEURONS_MLP_SQUASH_2_2014 = 32
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20

# My Parameters
GRU_RECURRENT_DROPOUT = 0.3
def stack(probs):
    return tf.stack(probs, axis=1)

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

def binary_accuracy(y_true, y_pred):
    r = K.print_tensor(K.round(y_pred), "ROund = ")
    return K.mean(K.equal(y_true, r), axis=-1)

def get_norm(vector):
    # n = K.print_tensor(norm(vector, ord=2, axis=1), "Norm of last vector = ")
    # return n
    return norm(vector, ord=2, axis=1)

def make_model(EMB_DIMENSIONS=300):
    # We don't need to specify the max length of the sequence, therefore None
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMB_DIMENSIONS, ))
    m = Bidirectional(GRU(128, dropout=DROPOUT,
                          recurrent_dropout=GRU_RECURRENT_DROPOUT,
                          return_sequences=False, unroll=True,
                          implementation=1))(sequence_input)
    m = Dropout(DROPOUT)(m)
    m = Dense(128, activation='relu', kernel_initializer=glorot_uniform())(m)
    m = Dropout(DROPOUT)(m)
    m = Dense(12, activation='sigmoid', kernel_initializer=glorot_uniform())(m)
    model = Model(sequence_input, m)
    opti = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)

    model.compile(optimizer=opti, loss='mean_squared_error', metrics=[metrics.binary_accuracy])
    return model


s = SemEvalData()
embedding = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
#embedding = Yelp(s.make_normal_vocabulary())
#x_train_val, y_train_val, x_test, y_test = s.get_x_embs_and_y_onehot(embedding)
x_train, y_train, validation = split_train_x_y_and_validation(0.1,
                                                            x_train_val, y_train_val)
model = make_model(len(x_train[0, 0, :]))
print(model.summary())
model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=50, validation_data=validation, callbacks=[get_early_stop_callback()])
pred_test = model.predict(x_test, batch_size=2)
e = 3
same_test_twice = np.array([x_test[-1, :, :], x_test[-1, :, :]])
#print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))


#Komn, Yelp: same predictions, always same majority class predicted