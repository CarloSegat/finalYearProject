import sklearn
import sys
sys.path.append('..\..\.')
import numpy as np
import random
from keras import Input, Model
from keras.layers import Embedding, concatenate, Dense, Flatten
from PolarityData import PolarityData
from embeddings.Embeddings import Komn, Yelp, Google
from utils import dump_gzip, load_gzip

#l = load_gzip('LSTM-ACD-Results')

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

def train(use_syntax, normal_sow_train, normal_sow_test, cat_train, cat_test,
          y_train, y_test, syntax_sow_train, syntax_sow_test, batch_size=64):

    def get_model(use_syntax):
        aspect_input = Input(shape=(1,))
        sentence_input = Input(shape=(len(normal_sow_train[0]), ))
        if use_syntax:
            syntax_input = Input(shape=(len(syntax_sow_train[0]), ))
            inputs = [sentence_input, syntax_input, aspect_input]
        else:
            inputs = [sentence_input, aspect_input]

        if use_syntax:
            full_sentence = concatenate([sentence_input, syntax_input])
        else:
            full_sentence = sentence_input

        asp_embedding = Embedding(input_dim=12,
                                output_dim=300,
                                trainable=True)(aspect_input)
        asp_embedding = Flatten()(asp_embedding)
        all_input = concatenate([full_sentence, asp_embedding])
        dense = Dense(100, activation='relu')(all_input)
        dense = Dense(3, activation='sigmoid')(dense)

        model = Model(inputs, dense)
        model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='nadam')
        #print(model.summary())
        return model

    model = get_model(use_syntax)
    if use_syntax:
        model.fit([normal_sow_train, syntax_sow_train, cat_train], y_train, epochs=1, batch_size=batch_size)
        pred_test = model.predict([normal_sow_test, syntax_sow_test, cat_test], batch_size=batch_size)
    else:
        model.fit([normal_sow_train, cat_train], y_train, epochs=1, batch_size=batch_size)
        pred_test = model.predict([normal_sow_test, cat_test], batch_size=batch_size)

    pred_test = np.argmax(pred_test, axis=1)
    pred_test = np.eye(3)[pred_test]
    return sklearn.metrics.f1_score(y_test, pred_test, average='micro')

# Data
data = PolarityData()
k = Komn(data.make_normal_vocabulary(), data.make_syntactical_vocabulary())
cat_train, cat_test = data.get_aspects_train_test(k)
syntax_sow_train, syntax_sow_test = data.get_x_train_test_syntax_polarity_sow(k)
syntax_sow_train, syntax_sow_test = syntax_sow_train[:, 300:600], syntax_sow_test[:, 300:600]

y_train, y_test = data.get_y_train_test_polarity()

# Embs
yelp = Yelp(data.make_normal_vocabulary())
komn = Komn(data.make_normal_vocabulary(), data.make_syntactical_vocabulary())
google = Google(data.make_normal_vocabulary())

batch_sizes = [16, 32, 64]

for p, punct in [(True, 'punct-removed'), (False, 'punct-kept')]:
    print(punct)
    for s, stop in [(True, 'stop-removed'), (False, 'stop-kept')]:
        print(stop)
        for embedding, emb in [(yelp, 'yelp'), (komn, 'komn'), (google, 'google')]:
            print(emb)
            normal_sow_train, normal_sow_test = data.get_normal_sentences_sow(embedding, p, s)

            for syntax, synt in [(True, 'synt'), (False, 'no-synt')]:
                print(synt)
                f1s = []
                for i in range(1):
                    batch_size = batch_sizes[random.randint(0, 2)]
                    f1s.append(train(syntax, normal_sow_train, normal_sow_test, cat_train, cat_test,
                                     y_train, y_test, syntax_sow_train, syntax_sow_test, batch_size=batch_size))

                all_scores[emb][synt][stop][punct] = f1s

dump_gzip(all_scores, 'FF-Polarity-Results')


