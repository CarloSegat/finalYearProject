import sklearn

from keras.callbacks import ModelCheckpoint

from Embeddings import Komn, Glove, Google, Yelp
from SemEval import get_data, SemEvalData
import keras as K
from keras.layers import Dense, regularizers, Softmax
from loss import micro_f1, cat_crossentropy_from_logit
from sklearn.metrics import f1_score


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

s = SemEvalData()
#Komn 0.47
#Google 0.49
#embs = Glove(300, s.make_vocabulary())
# yelp 0.52
# yelp on original ocde give 0.74 f1
embs = Yelp(s.make_vocabulary())
x, y, x_test, y_test = s.get_x_sow_and_y_onehot(embs)
model = get_model(sentence_embedding_length=len(x[0]))
model.load_weights(parameters_path)
#model.fit(x, y, batch_size=80, epochs=400, validation_split=0.15, callbacks=[checkpoint])
pred_test = model.predict(x_test, batch_size=80)
pred_test = pred_test > threshold
pred_test = pred_test + 0
#print(sklearn.metrics.f1_score(y, pred, average='micro'))
print(sklearn.metrics.f1_score(y_test, pred_test, average='micro'))