import sklearn
from keras.callbacks import ModelCheckpoint

from ACDData import ACDData
from embeddings.Embeddings import Komn
from SemEval import  SemEvalData
import keras as K
from keras.layers import Dense, regularizers
from loss import cat_crossentropy_from_logit
from sklearn.metrics import confusion_matrix
using_syntax = True
epochs = 1

def get_model(sentence_embedding_length=400, number_classes=12):
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
    opti = K.optimizers.adam(lr=0.002, decay=0.001)
    model.compile(optimizer=opti, loss=cat_crossentropy_from_logit, metrics=[])
    print(model.summary())
    return model

threshold = 0.785
parameters_path = "weigths.hdf5"
checkpoint = ModelCheckpoint(parameters_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

s = ACDData()
#Komn 0.6453 (different predictions)
#Google 0.69 (different predictions)
#Glove300  0.368 (same predictions)
# yelp 0.74 (different predictions)


embs = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
#embs = Google(s.make_normal_vocabulary())
#embs = Glove(300, s.make_normal_vocabulary())
#embs = Yelp(s.make_normal_vocabulary())

if using_syntax:
    x_train_val, y_train_val, x_test, y_test = s.get_data_syntax_concatenation_sow(embs)
else:
    x_train_val, y_train_val, x_test, y_test = s.get_x_sow_and_y_onehot(embs)
model = get_model(sentence_embedding_length=len(x_train_val[0]))

model.fit(x_train_val, y_train_val, batch_size=80, epochs=epochs, validation_split=0.15, callbacks=[checkpoint])

pred_test = model.predict(x_test, batch_size=80)
pred_test = pred_test > threshold
pred_test = pred_test + 0
print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))