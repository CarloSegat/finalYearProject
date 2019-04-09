import sklearn

from keras import Input, Sequential, Model
from keras.activations import softmax
from keras.backend import l2_normalize, dot
from keras.engine import Layer
from keras.initializers import glorot_uniform
from keras.layers import Dense, Dropout, Bidirectional, GRU, Lambda, Concatenate, concatenate, Softmax
from keras.optimizers import adam, SGD
from keras import backend as K
from keras.regularizers import Regularizer
from tensorflow import multiply, add, reduce_sum, matmul, transpose, reshape, eye, norm, tile
import tensorflow as tf
import numpy as np
from Embeddings import Komn
from SemEval import SemEvalData


# Paper Parameters
EMB_DIMENSIONS = 300
VALIDATION_PERCENTAGE = 0.1
TOPICS = 12
DROPOUT = 0.6
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
GRU_RECURRENT_DROPOUT = 0.2
def stack(probs):
    return tf.stack(probs, axis=1)

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

def get_norm(vector):
    return norm(vector, ord=2, axis=1)

def paper_regulariser(weights):
    normalised_weights = weights / tf.sqrt(tf.reduce_sum(tf.square(weights), axis=0, keepdims=True))
    dot_prod_between_topic_matrices = tf.matmul(tf.transpose(normalised_weights), normalised_weights)
    dot_prod_between_topic_matrices = K.print_tensor(dot_prod_between_topic_matrices,
                   "dot_prod_between_topic_matrices = ")
    minus_identity_matrix = dot_prod_between_topic_matrices - tf.eye(12)
    absolute_value = tf.abs(minus_identity_matrix)
    sum_columns = reduce_sum(absolute_value, axis=0)
    sum_all_elements = reduce_sum(sum_columns )
    return sum_all_elements

class My_Attention(Layer):

    def __init__(self, topics=12, **kwargs):
        self.topics = topics
        self.output_dim = 256
        super(My_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.batch_size = input_shape[0]
        self.kernel = self.add_weight(name='topic_vectors',
                                      shape=(256, self.topics),
                                      initializer=glorot_uniform(),
                                      trainable=True,
                                      regularizer=paper_regulariser)
        super(My_Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        reshaped_x = reshape(x, (-1, 256))
        importance_all_words_for_all_topics = matmul(reshaped_x, self.kernel)
        # All word embeddigns multiplied with all topic embeddings
        vs = []
        for t in range(12):
            importance_all_words_topic_t = reshape(importance_all_words_for_all_topics[:, t],
                                                   (80, -1))

            topic_weight_multiplier = tile(reshape(importance_all_words_topic_t,
                                                   (80 * tf.shape(x)[0], 1)), (1, 256))
            topic_weight_multiplier = softmax(topic_weight_multiplier, axis=0)
            r = topic_weight_multiplier * reshaped_x
            r = reshape(r, (80, -1))
            sum = reduce_sum(r, axis=0)
            v_on_rows = reshape(sum, (tf.shape(x)[0], 256))
            vs.append(v_on_rows)
        return vs

    def compute_output_shape(self, input_shape):
        return [(None, input_shape[2]) for i in range(12)]

def make_model():
    # We don't need to specify the max length of the sequence, therefore None
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMB_DIMENSIONS, ))
    m = Bidirectional(GRU(128, dropout=DROPOUT,
                          recurrent_dropout=GRU_RECURRENT_DROPOUT,
                          return_sequences=True, unroll=True,
                          implementation=2))(sequence_input)
    m = My_Attention()(m)
    topic_squash = []
    for i in range(TOPICS):
        d = Dense(NEURONS_MLP_SQUASH_1_2016, activation='tanh', kernel_initializer=glorot_uniform())(m[i])
        d = Dropout(DROPOUT)(d)
        d = Lambda(squash)(d)
        topic_squash.append(d)
    m = concatenate((topic_squash), axis=1)
    assert(m.shape[1] == NEURONS_MLP_SQUASH_1_2016 * TOPICS)
    p = []
    for i in range(TOPICS):
        d = Dense(NEURONS_MLP_SQUASH_2_2016, activation='tanh', kernel_initializer=glorot_uniform())(m)
        d = Dropout(DROPOUT)(d)
        d = Lambda(squash)(d)
        d = Lambda(get_norm)(d)
        p.append(d)
    m = Lambda(stack)(p)
    model = Model(sequence_input, m)
    opti = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)#adam()

    model.compile(optimizer=opti, loss='mean_squared_error', metrics=[])
    return model


s = SemEvalData()
embedding = Komn(s.make_vocabulary())
x_train, y_train, x_test, y_test = s.get_x_embs_and_y_onehot(embedding)
model = make_model()
print(model.summary())
model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=100, validation_split=VALIDATION_PERCENTAGE)
pred_test = model.predict(x_test)
print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))