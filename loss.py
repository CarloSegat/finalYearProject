import sklearn
from keras import backend as K
from keras.backend import categorical_crossentropy
import tensorflow as tf

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    This lets you apply a weight to unbalanced classes.

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss.py = weighted_categorical_crossentropy(weights)
        model.compile(loss.py=loss.py,optimizer='adam')

    We pick the cross-entropy as the loss function, and we weight it by the inverse
    frequency of the true classes to counteract the imbalanced dataset.


    y_pred = [[0.135464132 0.261906356 0.602629542]...]
    y_true = [[1 0 0]...]

    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        #y_true = K.print_tensor(y_true, message='y_true = ')
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

def f1_score(true,pred):
    pred = K.cast(K.greater(pred,0.5), K.floatx())

    groundPositives = K.sum(true) + K.epsilon()
    correctPositives = K.sum(true * pred) + K.epsilon()
    predictedPositives = K.sum(pred) + K.epsilon()

    precision = correctPositives / predictedPositives
    recall = correctPositives / groundPositives

    m = (2 * precision * recall) / (precision + recall)

    return m

def micro_f1(y_true, y_pred):
    '''
    https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras
    '''
    return sklearn.metrics.f1_score(y_true, y_pred, average='micro')

def cat_crossentropy_from_logit(target, output,):
    return categorical_crossentropy(target, output, from_logits=True)

def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma)
                  * K.log(pt_1))-K.sum((1-alpha)
                  * K.pow( pt_0, gamma) * K.log(1. - pt_0))
