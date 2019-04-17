from keras import Input, Model
from keras.layers import Embedding, concatenate, Dense, Flatten
from keras.optimizers import nadam

from PolarityData import PolarityData
from embeddings.Embeddings import Komn

#Params
use_syntax = False

# Data
p = PolarityData()
k = Komn(p.make_normal_vocabulary(), p.make_syntactical_vocabulary())
syntax_sow_train, syntax_sow_test = p.get_x_train_test_syntax_polarity_sow(k)
syntax_sow_train, syntax_sow_test = syntax_sow_train[:, 300:600], syntax_sow_test[:, 300:600]
normal_sow_train, normal_sow_test = p.get_normal_sentences_sow(k)
cat_train, cat_test = p.get_aspects_train_test(k)
y_train, y_test = p.get_y_train_test_polarity()

# Net
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
                        output_dim=12,
                        trainable=True)(aspect_input)
asp_embedding = Flatten()(asp_embedding)
all_input = concatenate([full_sentence, asp_embedding])
dense = Dense(100, activation='relu')(all_input)
dense = Dense(3, activation='sigmoid')(dense)

opti = nadam()


model = Model(inputs, dense)
model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer=opti)
print(model.summary())

#Train
if use_syntax:
    model.fit([normal_sow_train, syntax_sow_train, cat_train], y_train)
else:
    model.fit([normal_sow_train, cat_train], y_train)