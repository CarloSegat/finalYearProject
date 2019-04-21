import sklearn
import sys

from utils import dump_gzip

sys.path.append('..\..\.')
from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Flatten, RepeatVector, concatenate, LSTM, Lambda, multiply, Dense, \
    Activation, add
import keras.backend as K
from keras.optimizers import nadam
import numpy as np

from PolarityData import PolarityData
from SemEval import SemEvalData
from custom_layers import Attention
from embeddings.Embeddings import Komn, Yelp, Google

all_scores = {'komn':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                     'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
             'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                        'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
            },

             'google':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    },
            'yelp':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                             'stop-removed':{'punct-kept':[], 'punct-removed':[]}},
                     'no-synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
                                'stop-removed':{'punct-kept':[], 'punct-removed':[]}}
                    }
            }

def train(use_syntax, w, x_train_val, x_test, x_syntax_train, x_syntax_test,
          y_train, y_test, batch_size, epochs):


    def get_model(use_syntax):
        n_aspect = 12
        lstm_units = 100
        TRAINABLE_EMBS = False
        TRAINABLE_ASPECTS = False
        input_text = Input(shape=(80,))
        input_aspect = Input(shape=(1,), )
        if use_syntax:
            input_syntax = Input(shape=(80,300,))

        word_embedding = Embedding(input_dim=len(w), output_dim=len(w[0]),
                                    trainable=TRAINABLE_EMBS,
                                    mask_zero=True,
                                    weights=[w])(input_text)
        text_embed = SpatialDropout1D(0.2)(word_embedding)
        if use_syntax:
            text_embed = concatenate([text_embed, input_syntax])

        asp_embedding = Embedding(input_dim=n_aspect,
                                output_dim=100,
                                trainable=TRAINABLE_ASPECTS)
        aspect_embed = asp_embedding(input_aspect)
        aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
        repeat_aspect = RepeatVector(80)(aspect_embed)  # repeat aspect for every word in sequence

        input_concat = concatenate([text_embed, repeat_aspect], axis=-1)
        hidden_vecs, state_h, _ = LSTM(lstm_units, return_sequences=True, return_state=True)(input_concat)
        concat = concatenate([hidden_vecs, repeat_aspect], axis=-1)

        # apply attention mechanism
        attend_weight = Attention()(concat)
        attend_weight_expand = Lambda(lambda x: K.expand_dims(x))(attend_weight)
        attend_hidden = multiply([hidden_vecs, attend_weight_expand])
        attend_hidden = Lambda(lambda x: K.sum(x, axis=1))(attend_hidden)

        attend_hidden_dense = Dense(lstm_units)(attend_hidden)
        last_hidden_dense = Dense(lstm_units)(state_h)
        final_output = Activation('tanh')(add([attend_hidden_dense, last_hidden_dense]))


        if use_syntax:
            base_network = Model([input_text, input_aspect, input_syntax], final_output)
        else:
            base_network = Model([input_text, input_aspect], final_output)

        network_inputs = [Input(shape=(80,), name='sentence'),
                          Input(shape=(1, ), name='input_aspect')]
        if use_syntax:
            network_inputs.append(input_syntax)

        sentence_vec = base_network(network_inputs)
        dense_layer = Dense(30, activation='relu')(sentence_vec)
        output_layer = Dense(3, activation='softmax')(dense_layer)

        model = Model(network_inputs, output_layer)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='nadam')
        return model

    #fake_x = [np.array([[1]*80]), np.array([1])]
    #fake_y = np.array([[1, 0, 0]])

    model = get_model(use_syntax)
    if use_syntax:
        model.fit([x_train_val[0], x_train_val[1], x_syntax_train],
                  y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.15,)
        pred_test = model.predict([x_test[0], x_test[1], x_syntax_test],
                                  batch_size=batch_size)
    else:
        model.fit([x_train_val[0], x_train_val[1]],
                  y_train, epochs=epochs, batch_size=batch_size,
                  validation_split=0.15,)
        pred_test = model.predict([x_test[0], x_test[1]],
                                  batch_size=batch_size)

    pred_test = np.argmax(pred_test, axis=1)
    pred_test = np.eye(3)[pred_test]
    return sklearn.metrics.f1_score(y_test, pred_test, average='micro')

# Data
s = PolarityData()
embs = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
x_train_val, y_train_val, x_test, y_test, w = s.get_data_as_integers_and_emb_weights_polarity(embs)


# Data
p = PolarityData()

k = Komn(p.make_normal_vocabulary(), p.make_syntactical_vocabulary())
x_syntax_train, x_syntax_test = p.get_x_train_test_syntax_polarity(k, pad=True)
x_syntax_train, x_syntax_test = x_syntax_train[:, :, 300:600], x_syntax_test[:, :, 300:600]


# Embs
yelp = Yelp(p.make_normal_vocabulary())
komn = Komn(p.make_normal_vocabulary(), p.make_syntactical_vocabulary())
google = Google(p.make_normal_vocabulary())

batch_sizes = [16, 32, 64]



for pun, punct in [(True, 'punct-removed'), (False, 'punct-kept')]:
    print(punct)
    for s, stop in [(True, 'stop-removed'), (False, 'stop-kept')]:
        print(stop)
        for embedding, emb in [(yelp, 'yelp'), (komn, 'komn'), (google, 'google')]:
            print(emb)
            x_train, y_train, x_test, y_test, w = \
                p.get_data_as_integers_and_emb_weights_polarity(embedding, s, pun)


            for syntax, synt in [(True, 'synt'), (False, 'no-synt')]:
                print(synt)
                f1s = []
                for i in range(1):
                    f1s.append(train(syntax, w, x_train, x_test, x_syntax_train, x_syntax_test,
                                     y_train, y_test, epochs=1, batch_size=200))
                all_scores[emb][synt][stop][punct] = f1s

dump_gzip(all_scores, 'ATAE-LSTM-Polarity-Results')


#F1 Scores:

# using syntax: batch 64, micro averaged, 10 epochs: 0.734

# without syntax: batch 64, micro averaged, 10 epochs: 0.748