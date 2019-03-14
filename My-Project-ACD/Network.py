import keras as K
from keras.layers import Dense, regularizers, Softmax


def get_model(vocabulary_size, max_sentence_length, embedding_vector_length=400,
        lstm_dropout=0.2, lstm_recurrent_dropout=0.2, number_classes=12):
        model = K.models.Sequential()
        dense_output_dim = number_classes
        model.add(Dense(250,
                        input_shape=(embedding_vector_length,),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(number_classes,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)))
        model.add(Softmax())
        model.compile(optimizer='adam', loss='categorical_cross_entropy')
        return model