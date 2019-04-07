import keras as K
from keras.layers import Bidirectional, LSTM, Dense, regularizers, Softmax


def create_model(vocabulary_size, max_sentence_length, embedding_vector_length=300,
                 lstm_dropout=0.2, lstm_recurrent_dropout=0.2, number_classes=3):
    model = K.models.Sequential()
    model.add(K.layers.embeddings.Embedding(input_dim=vocabulary_size,
                                            output_dim=embedding_vector_length,
                                            mask_zero=True))
    lstm_units = max_sentence_length
    model.add(Bidirectional(LSTM(lstm_units,
                                 return_sequences=True,
                                 dropout=lstm_dropout,
                                 recurrent_dropout=lstm_recurrent_dropout)))
    dense_output_dim = number_classes
    model.add(Dense(dense_output_dim,
                    activation='linear',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Softmax())