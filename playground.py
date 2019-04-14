import numpy as np
from sklearn.utils.class_weight import compute_class_weight

y_integers = np.argmax(np.array([[1, 0, 1], [0, 1, 0],
                                 [1, 0, 1], [0, 0, 1], [0, 1, 0],
                                 [1, 1, 0], [0, 0, 1], [0, 1, 0],
                                 [1, 1, 0], [1, 1, 0], [1, 1, 0]]), axis=1)
class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
d_class_weights = dict(enumerate(class_weights))

from keras.preprocessing.text import Tokenizer
texts = ['a a a', 'b b', 'c']
tokenizer = Tokenizer(num_words=2)
tokenizer.fit_on_texts(texts)
wd = tokenizer.word_index
tokenizer.texts_to_sequences(texts)
