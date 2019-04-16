from keras_preprocessing.sequence import pad_sequences

from SemEval import SemEvalData, ASPECT_CATEGORIES, TRAIN_SENTENCES
import numpy as np

from utils import pad_array


class ACDData(SemEvalData):

    def __init__(self):
        super(ACDData, self).__init__()

    def get_max_len(self, sequences):
        lengths = []
        for x in sequences:
            if not hasattr(x, '__len__'):
                raise ValueError('`sequences` must be a list of iterables. '
                                 'Found non-iterable: ' + str(x))
            lengths.append(len(x))
        return np.max(lengths)

    def make_multilabel_1hot_vector(self, aspect_categories):
        aspect_categories = list(set(aspect_categories))
        multiclass_label = np.zeros(12)
        for ac in aspect_categories:
            multiclass_label = np.add(multiclass_label, ASPECT_CATEGORIES[ac])
        return multiclass_label

    def get_data_as_integers_and_emb_weights(self, embbedings, pad=True):
        '''Returns sentences converted using word indices and also the
        weights to put intot he embedding layer to get the conversion'''

        def get_x_y(trained_tagged, trained_normal, word_to_int):
            X, Y = [], []
            for x, xx in zip(trained_tagged, trained_normal):
                build = []
                for w in x[0]:
                    build.append(word_to_int[w])
                X.append(build)

                cats = []
                for opinion in trained_normal[xx]['opinions']:
                    cats.append(opinion['category'])
                Y.append(self.make_multilabel_1hot_vector(cats))
            max_len = self.get_max_len(X)
            L = (80 - max_len) // 2
            X = pad_sequences(X, maxlen=max_len + L, padding='pre')
            X = pad_sequences(X, maxlen=80, padding='post')
            return X, np.array(Y)

        word_to_int = {v: k for k, v in enumerate(self.make_normal_vocabulary())}
        int_to_word = {k: v for k, v in enumerate(self.make_normal_vocabulary())}
        X_train, Y_train = get_x_y(self.ready_tagged_train, self.ready_train, word_to_int)
        X_test, Y_test = get_x_y(self.ready_tagged_test, self.ready_test, word_to_int)

        weights = []
        for i in range(len(int_to_word)):
            try:
                weights.append(embbedings.word_to_emb[int_to_word[i]])
            except Exception:
                weights.append(embbedings.default_emb)
        return X_train, Y_train, X_test, Y_test, np.array(weights)


    def get_x_train_test_syntax(self, komn, pad=False):
        x_train, x_test = [], []
        for i in range(len(self.ready_tagged)):
            sc = komn.get_syntactic_concatenation(self.ready_tagged[i])
            if pad:
                sc = pad_array(sc, 80)
            if i < TRAIN_SENTENCES:
                x_train.append(sc)
            else:
                x_test.append(sc)
        return np.array(x_train),np.array(x_test)

    def get_data_syntax_concatenation(self, komn):
        x_train, x_test = self.get_x_train_test_syntax(komn, pad=True)
        _, y_train, _, y_test, _ = self.get_data_as_integers_and_emb_weights(komn)
        return x_train, y_train, x_test, y_test