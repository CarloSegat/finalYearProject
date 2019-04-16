import sklearn
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Softmax, regularizers, Bidirectional, LSTM, Dropout, Embedding, concatenate, \
    Convolution1D, MaxPooling1D, Flatten, GlobalMaxPooling1D
from keras.optimizers import adam
from ACDData import ACDData
from embeddings.Embeddings import Komn
from utils import get_early_stop_callback, split_train_x_y_and_validation
from sklearn.metrics import confusion_matrix

use_syntax = False
epochs = 1

def save_all(models):
    for i, m in enumerate(models):
        m.save("class" + str(i))

def get_all_predictions(models, x_test):
    all_preds = np.zeros_like(y_test)
    for i, m in enumerate(models):
        pred = models[i].predict(x_test, batch_size=batch_size)
        pred = pred > 0.5
        all_preds[:, i] = np.squeeze(pred)
        # print(sklearn.metrics.f1_score(y_test[:, i], pred))

def create_model(max_sentence_length, w, TRAINABLE_EMBS=False,
                 lstm_dropout=0.2, lstm_recurrent_dropout=0.2):
    input_text = Input(shape=(max_sentence_length,))
    embs = Embedding(input_dim=len(w), output_dim=len(w[0]),
                            trainable=TRAINABLE_EMBS,
                            weights=[w])(input_text)
    embs = Dropout(0.2)(embs)
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
        #conv = Flatten()(conv)
        conv_blocks.append(conv)
    convolu = concatenate(conv_blocks)
    lstm = Dropout(0.2)(convolu)
    dense = Dense(1,activation='sigmoid',
                    kernel_regularizer=regularizers.l2())(lstm)

    network_inputs = [input_text]
    if use_syntax:
        network_inputs.append(syntax_input)

    model = Model(network_inputs, dense)
    opti = adam(lr=0.001, clipvalue=1.0)
    model.compile(optimizer=opti, loss='binary_crossentropy')
    print(model.summary())
    return model

s = ACDData()
embs = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
if use_syntax:
    x_sytax_train, _, x_sytax_test, _ = s.get_data_syntax_concatenation(embs)
    x_sytax_train, x_sytax_test = x_sytax_train[:, :, 300:600], x_sytax_test[:, :, 300:600]


x_train_val, y_train_val, x_test, y_test, w = s.get_data_as_integers_and_emb_weights(embs)

batch_size = 128
models = [None]*12

x_train_val = [x_train_val]
x_test = [x_test]
if use_syntax:
    x_train_val.append(x_sytax_train)
    x_test.append(x_sytax_test)

for i in range(12):
    models[i] = create_model(80,w)
    models[i].fit(x_train_val, y_train_val[:, i], batch_size=batch_size,
          epochs=10, validation_split=0.15, shuffle=True,
          callbacks=[get_early_stop_callback()])
pred = models[2].predict(x_test, batch_size=batch_size)
pred = pred > 0.1
sum(pred)
print(sklearn.metrics.f1_score(y_test, pred, average='binary'))

# f1 0.5206 binary_crossentropy, softamx, 5 epochs, batch=32
# f1 0.6307 binary_crossentropy, softmax, 10 epochs,, batch=32
# f1 0.67 b_c,sm, 100 epochs, batch=128, only 2 classes haven't been predicted ever
# sigmoid as last layer (no osftamx) doesnt work (same class predicted)