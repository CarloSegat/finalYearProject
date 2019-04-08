from keras import Input, Sequential, Model
from keras.activations import softmax
from keras.engine import Layer
from keras.layers import Dense, Dropout, Bidirectional, GRU
from keras.optimizers import adam
from keras import backend as K
from tensorflow import multiply, add, reduce_sum


class My_Attention(Layer):

    def __init__(self, **kwargs):
        self.output_dim = 256
        super(My_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert(input_shape[1] == self.output_dim)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='topic_vector',
                                      shape=(input_shape[1], 1),
                                      initializer='uniform',
                                      trainable=True)
        super(My_Attention, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dots = K.dot(x, self.kernel)
        alphas = softmax(dots)
        v = reduce_sum(multiply(x, alphas), 0)
        return v

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

# Paper Parameters
EMB_DIMENSIONS = 300
VALIDATION_PERCENTAGE = 0.1
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

# We don't need to specify the max length of the sequence, therefore None
sequence_input = Input(shape=(None, EMB_DIMENSIONS,))
m = Bidirectional(GRU(128, dropout=DROPOUT,
                  recurrent_dropout=GRU_RECURRENT_DROPOUT))(sequence_input)
m = My_Attention()(m)
model = Model(sequence_input, m)
opti = adam(lr=0.002, decay=0.001)

model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=[])
print(model.summary())
