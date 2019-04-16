import sklearn

from keras import Input, Model
from keras.layers import Embedding, SpatialDropout1D, Flatten, RepeatVector, concatenate, LSTM, Lambda, multiply, Dense, \
    Activation, add
import keras.backend as K
from keras.optimizers import nadam
import numpy as np
from SemEval import SemEvalData
from custom_layers import Attention
from embeddings.Embeddings import Komn

s = SemEvalData()
embs = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
x_train_val, y_train_val, x_test, y_test, w = s.get_data_as_integers_and_emb_weights_polarity(embs)


use_syntax = False
MAX_LEN = 80
if use_syntax:
    VOC_SIZE = len(w)
    EMB_DIM = len(list(embs.syntax_word_to_emb.values())[0])
else:
    VOC_SIZE = len(w)
    EMB_DIM = len(list(embs.word_to_emb.values())[0])

TRAINABLE_EMBS = False
TRAINABLE_ASPECTS = False
aspect_embed_type = 'random'
n_aspect = 12
lstm_units = 100

input_text = Input(shape=(MAX_LEN,))
input_aspect = Input(shape=(1,), )
if use_syntax:
    # Those arrays need to be concatenated as they include syntactic information
    x_sytax_train, x_sytax_test = s.get_x_train_test_syntax_for_polarity(embs)
    x_sytax_train, x_sytax_test = x_sytax_train[:, :, 300:600], x_sytax_test[:, :, 300:600]
    input_syntax = Input(shape=(80,300,))


# z = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding")(model_input)

word_embedding = Embedding(input_dim=VOC_SIZE, output_dim=EMB_DIM,
                            trainable=TRAINABLE_EMBS,
                            mask_zero=True,
                            weights=[w])(input_text)
text_embed = SpatialDropout1D(0.2)(word_embedding)
if use_syntax:
    text_embed = concatenate([text_embed, input_syntax])

asp_embedding = Embedding(input_dim=n_aspect,
                        output_dim=EMB_DIM,
                        trainable=TRAINABLE_ASPECTS)
aspect_embed = asp_embedding(input_aspect)
aspect_embed = Flatten()(aspect_embed)  # reshape to 2d
repeat_aspect = RepeatVector(MAX_LEN)(aspect_embed)  # repeat aspect for every word in sequence

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

network_inputs = [Input(shape=(MAX_LEN,), name='sentence'),
                  Input(shape=(1, ), name='input_aspect')]
if use_syntax:
    network_inputs.append(input_syntax)

sentence_vec = base_network(network_inputs)
dense_layer = Dense(30, activation='relu')(sentence_vec)
output_layer = Dense(3, activation='softmax')(dense_layer)

opti = nadam()

model = Model(network_inputs, output_layer)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opti)

#fake_x = [np.array([[1]*80]), np.array([1])]
#fake_y = np.array([[1, 0, 0]])
if use_syntax:
    x_train_val.append(x_sytax_train)
    x_test.append(x_sytax_test)

model.fit(x_train_val, y_train_val, epochs=10, validation_split=0.15, batch_size=64)
#print(model.summary())
pred_test = model.predict(x_test, batch_size=64)
pred_test = np.argmax(pred_test, axis=1)
pred_test = np.eye(3)[pred_test]
print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))

a = 3

#F1 Scores:

# using syntax: batch 64, micro averaged, 10 epochs: 0.734

# without syntax: batch 64, micro averaged, 10 epochs: 0.748