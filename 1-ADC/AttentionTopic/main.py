from keras import Input,  Model, metrics
from keras.activations import softmax
from keras.engine import Layer
from keras.initializers import glorot_uniform, glorot_normal
from keras.layers import Dense, Dropout, Bidirectional, GRU, Lambda, concatenate
from keras.optimizers import SGD
from keras import backend as K
from tensorflow import reduce_sum, matmul, reshape, norm, tile
import tensorflow as tf
import numpy as np
from embeddings.Embeddings import Yelp
from SemEval import SemEvalData
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
    '''Makes a vector length between 0 and 1'''
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
    return scale * vectors

def get_norm(vector):
    # n = K.print_tensor(norm(vector, ord=2, axis=1), "Norm of last vector = ")
    # return n
    return norm(vector, ord=2, axis=1)

def paper_regulariser(weights):
    '''Purpose is to make topic vectors perpendicular, i.e. their
    dot product should be zero'''
    normalised_weights = weights / tf.sqrt(tf.reduce_sum(tf.square(weights), axis=0, keepdims=True))
    dot_prod_between_topic_matrices = tf.matmul(tf.transpose(normalised_weights), normalised_weights)
    dot_prod_between_topic_matrices = K.print_tensor(dot_prod_between_topic_matrices,
                   "dot_prod_between_topic_matrices = ")
    minus_identity_matrix = dot_prod_between_topic_matrices - tf.eye(11)
    absolute_value = tf.abs(minus_identity_matrix)
    sum_columns = reduce_sum(absolute_value, axis=0)
    sum_all_elements = reduce_sum(sum_columns )
    return sum_all_elements

class My_Attention(Layer):

    def __init__(self, topics=11, **kwargs):
        self.topics = topics
        self.output_dim = 256
        super(My_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.batch_size = input_shape[0]
        self.kernel = self.add_weight(name='topic_vectors',
                                      shape=(256, self.topics),
                                      initializer=glorot_normal(),
                                      trainable=True,
                                      regularizer=paper_regulariser)
        super(My_Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        reshaped_x = reshape(x, (-1, 256))
        importance_all_words_for_all_topics = matmul(reshaped_x, self.kernel)
        # All word embeddigns multiplied with all topic embeddings
        vs = []
        for t in range(self.topics):
            importance_all_words_topic_t = reshape(importance_all_words_for_all_topics[:, t],
                                                   (80, -1))

            importance_all_words_topic_t = softmax(importance_all_words_topic_t, axis=0)
            topic_weight_multiplier = tile(reshape(importance_all_words_topic_t,
                                                   (80 * tf.shape(x)[0], 1)), (1, 256))
            r = topic_weight_multiplier * reshaped_x
            r = reshape(r, (80, -1))
            sum = reduce_sum(r, axis=0, keep_dims=True)
            v_on_rows = reshape(sum, (tf.shape(x)[0], 256))
            vs.append(v_on_rows)
        return vs

    def compute_output_shape(self, input_shape):
        return [(None, input_shape[2]) for i in range(self.topics)]

def make_model(EMB_DIMENSIONS=300):
    # We don't need to specify the max length of the sequence, therefore None
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, EMB_DIMENSIONS, ))
    m = Bidirectional(GRU(128, dropout=DROPOUT,
                          recurrent_dropout=GRU_RECURRENT_DROPOUT,
                          return_sequences=True, unroll=True,
                          implementation=1))(sequence_input)
    m = My_Attention()(m)
    topic_squash = []
    for i in range(TOPICS):
        d = Dropout(DROPOUT)(m[i])
        d = Dense(NEURONS_MLP_SQUASH_1_2016, activation=None,
                  kernel_initializer=glorot_uniform())(d)
        d = Lambda(squash)(d)
        topic_squash.append(d)
    m = concatenate((topic_squash), axis=1)
    assert(m.shape[1] == NEURONS_MLP_SQUASH_1_2016 * TOPICS)
    p = []
    for i in range(TOPICS+1):
        m = Dropout(DROPOUT)(m)
        d = Dense(NEURONS_MLP_SQUASH_2_2016, activation=None, kernel_initializer=glorot_uniform())(m)
        d = Lambda(squash)(d)
        d = Lambda(get_norm)(d)
        p.append(d)
    #m = Concatenate(p)
    m = Lambda(stack)(p)
    model = Model(sequence_input, m)
    opti = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipvalue=0.5)

    model.compile(optimizer=opti, loss='binary_crossentropy', metrics=[metrics.binary_accuracy])
    return model


s = SemEvalData()
#embedding = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
embedding = Yelp(s.make_normal_vocabulary())
x_train_val, y_train_val, x_test, y_test = s.get_x_embs_and_y_onehot(embedding)
x_train, y_train, validation = split_train_x_y_and_validation(0.1,
                                                            x_train_val, y_train_val)

weights = dict(enumerate(max(sum(y_train)) / sum(y_train)))
model = make_model(len(x_train[0, 0, :]))
print(model.summary())
model.fit(x_train, y_train, batch_size=BATCH_SIZE,
          epochs=50, validation_data=validation,
          callbacks=[get_early_stop_callback()],
          class_weight=weights
          )
pred_test = model.predict(x_test)
e = 3
same_test_twice = np.array([x_test[-1, :, :], x_test[-1, :, :]])
#print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))

# Validation_loss of 0.3 with cross entropy
# V loss of o.15 with MSE

#Komn: same predictions, always same majority class predicted

# Yelp: class 5 has constant prediction score but then other class have randomly
# higher scores; seems arbitrary:classes from 7 to 11 have higher scores