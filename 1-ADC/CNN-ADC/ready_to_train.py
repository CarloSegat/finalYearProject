import random, sklearn, sys
sys.path.append('..\..\.')
from keras import Input, Model
from keras.layers import Dropout, concatenate, Convolution1D, GlobalMaxPooling1D, Dense, regularizers, Embedding, \
    SpatialDropout1D

from ACDData import ACDData
import numpy as np

from embeddings.Embeddings import Yelp, Komn, Google, Glove
from utils import dump_gzip, get_early_stop_callback

#l =load_gzip('CNN-ACD-Results-Komn')

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


def train(use_syntax, syntax_train, syntax_test, y_train_val, y_test,
          x_train, x_test, w, max_epochs=50, batch_size=32):

    def get_model(w, use_syntax, TRAINABLE_EMBS=False):
        input_text = Input(shape=(80,))
        embs = Embedding(input_dim=len(w), output_dim=len(w[0]),
                         trainable=TRAINABLE_EMBS,
                         weights=[w])(input_text)
        embs = SpatialDropout1D(0.25)(embs)
        if use_syntax:
            syntax_input = Input(shape=(80, 300,))
            embs = concatenate([embs, syntax_input])
        conv_blocks = []
        for sz in (2, 3, 5):
            conv = Convolution1D(filters=100,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(embs)
            conv = GlobalMaxPooling1D()(conv)
            # conv = Flatten()(conv)
            conv_blocks.append(conv)
        convolu = concatenate(conv_blocks)
        lstm = Dropout(0.25)(convolu)
        dense = Dense(1, activation='sigmoid',
                      kernel_regularizer=regularizers.l2())(lstm)

        network_inputs = [input_text]
        if use_syntax:
            network_inputs.append(syntax_input)

        model = Model(network_inputs, dense)
        model.compile(optimizer='nadam', loss='binary_crossentropy')
        #print(model.summary())
        return model

    x_train = [x_train]
    x_test = [x_test]

    if use_syntax:
        x_train.append(syntax_train)
        x_test.append(syntax_test)

    models = [get_model(w, use_syntax)]*12
    for i in range(12):
        models[i].fit(x_train, y_train_val[:, i], verbose=0, batch_size=batch_size,
                      epochs=max_epochs, validation_split=0.15,
                      callbacks=[get_early_stop_callback()])
        pred = models[i].predict(x_test, batch_size=batch_size)
        pred = pred > 0.1
        if i == 0:
            all_pred = pred
        else:
            all_pred = np.hstack((all_pred, pred))
    return sklearn.metrics.f1_score(y_test, all_pred, average='micro')

data = ACDData()

yelp = Yelp(data.make_normal_vocabulary())
komn = Komn(data.make_normal_vocabulary(), data.make_syntactical_vocabulary())
google = Google(data.make_normal_vocabulary())
glove = Glove(300, data.make_normal_vocabulary())

syntax_train, y_train_val, syntax_test, y_test = data.get_data_syntax_concatenation(komn)
syntax_train, syntax_test = syntax_train[:, :, 300:600], syntax_test[:, :, 300:600]

batch_sizes = [16, 32, 64]

for punctuation_removed, punct in [(True, 'punct-removed'), (False, 'punct-kept')]:
    print(punct)
    for stopwords_removed, stop in [(True, 'stop-removed'), (False, 'stop-kept')]:
        print(stop)
        for embedding, emb in [(google, 'google')]: # , (google, 'google'), (yelp, 'yelp'), (komn, 'komn')
            print(emb)
            for syntax, synt in [(True, 'synt'), (False, 'no-synt')]:
                print(synt)
                f1s = []
                x_train, _, x_test, _, w = data.get_data_as_integers_and_emb_weights(embedding,
                                                                no_stop=stopwords_removed,
                                                                no_punct=punctuation_removed)
                for i in range(10):
                    batch_size = batch_sizes[random.randint(0, 2)]
                    f1s.append(train(syntax, syntax_train, syntax_test, y_train_val, y_test, x_train, x_test, w,
                                     max_epochs=50,
                                     batch_size=batch_size))

                all_scores[emb][synt][stop][punct] = f1s

dump_gzip(all_scores, 'CNN-ACD-Results')