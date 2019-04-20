import random

import sklearn
import numpy as np
from ACDData import ACDData
from embeddings.Embeddings import Komn, Google, Yelp, Glove
import keras as K
from keras.layers import Dense, regularizers
from utils import get_early_stop_callback, dump_gzip

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

             'glove':{'synt':{'stop-kept':{'punct-kept':[], 'punct-removed':[]},
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
          x_train, x_test, max_epochs=200, batch_size=32):

    def get_model(sentence_embedding_length, number_classes=12):
        model = K.models.Sequential()
        model.add(Dense(300,
                        input_shape=(sentence_embedding_length,),
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.002)))
        model.add(Dense(250,
                        activation='relu',
                        kernel_regularizer=regularizers.l2(0.002)
                        ))
        model.add(Dense(number_classes,
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(0.002)))
        model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=[])
        #print(model.summary())
        return model

    threshold = 0.785
    #Komn 0.6453 (different predictions)
    #Google 0.69 (different predictions)
    #Glove300  0.368 (same predictions)
    # yelp 0.74 (different predictions)


    if use_syntax:
        x_train = np.hstack((x_train, syntax_train[:, 300:600]))
        x_test = np.hstack((x_test, syntax_test[:, 300:600]))

    model = get_model(sentence_embedding_length=len(x_train[0]))

    model.fit(x_train, y_train_val, batch_size=batch_size, epochs=max_epochs,
              validation_split=0.15, callbacks=[get_early_stop_callback()], verbose=0)

    pred_test = model.predict(x_test, batch_size=batch_size)
    pred_test = pred_test > threshold
    pred_test = pred_test + 0
    return sklearn.metrics.f1_score(y_test, pred_test, average='micro')

data = ACDData()

yelp = Yelp(data.make_normal_vocabulary())
komn = Komn(data.make_normal_vocabulary(), data.make_syntactical_vocabulary())
google = Google(data.make_normal_vocabulary())
glove = Glove(300, data.make_normal_vocabulary())

syntax_train, y_train_val, syntax_test, y_test = data.get_data_syntax_concatenation_sow(komn)

batch_sizes = [16, 32, 64]

for punctuation_removed, punct in [(True, 'punct-removed'), (False, 'punct-kept')]:
    print(punct)
    for stopwords_removed, stop in [(True, 'stop-removed'), (False, 'stop-kept')]:
        print(stop)
        for embedding, emb in [(yelp, 'yelp')]:  #(komn, 'komn'), (google, 'google')
            print(emb)
            for syntax, synt in [(True, 'synt'), (False, 'no-synt')]:
                f1s = []
                x_train, x_test = data.get_normal_sentences_sow(embedding,
                                                                no_stop=stopwords_removed,
                                                                no_punct=punctuation_removed)
                for i in range(10):
                    batch_size = batch_sizes[random.randint(0, 2)]
                    f1s.append(train(syntax, syntax_train, syntax_test, y_train_val, y_test, x_train, x_test,
                                     max_epochs=50,
                                     batch_size=batch_size))

                all_scores[emb][synt][stop][punct] = f1s

dump_gzip(all_scores, 'FF-ACD-Results')