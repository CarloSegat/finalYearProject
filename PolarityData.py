import numpy as np
from keras_preprocessing.sequence import pad_sequences

from SemEval import ASPECT_CATEGORIES, TRAIN_OPINIONS, TEST_OPINIONS, SemEvalData, TRAIN_SENTENCES
from utils import pad_array


class PolarityData(SemEvalData):

    def __init__(self):
        super(PolarityData, self).__init__()

    def get_y_train_test_polarity(self):
        p = {'positive': [1,0,0], 'neutral': [0,1,0], 'negative': [0,0,1]}
        Y_train, Y_test = [], []
        for s in self.ready_train:
            for opinion in self.ready_train[s]['opinions']:
                Y_train.append(p[opinion['polarity']])
        for s in self.ready_test:
            for opinion in self.ready_test[s]['opinions']:
                Y_test.append(p[opinion['polarity']])
        return np.array(Y_train), np.array(Y_test)

    def get_data_as_integers_and_emb_weights_polarity(self, embbedings, no_stop=False, no_punct=False):
        '''Returns sentences converted using word indices and also the
        weights to put intot he embedding layer to get the conversion'''

        def map_sentences_to_cat(output, trained_tagged, trained_normal, word_to_int):
            for x, xx in zip(trained_tagged, trained_normal):
                build = []
                s = x[0]

                if no_stop:
                    s = self.text_preprocessor.remove_stopwords(s)
                if no_punct:
                    s = self.text_preprocessor.remove_punctuation(s)

                for w in s:
                    build.append(word_to_int[w])


                for opinion in trained_normal[xx]['opinions']:
                    cat = ASPECT_CATEGORIES[opinion['category']]
                    cat = np.where(cat)[0][0]
                    output[0].append(build)
                    output[1].append(cat)
            output[0] = pad_sequences(output[0], maxlen=80)
            output[1] = np.array(output[1])

        word_to_int = {v:k for k, v in enumerate(self.make_normal_vocabulary())}
        int_to_word = {k: v for k, v in enumerate(self.make_normal_vocabulary())}
        X_train = [[],[]]
        X_test = [[],[]]
        Y_train , Y_test = self.get_y_train_test_polarity()
        map_sentences_to_cat(X_train, self.ready_tagged_train, self.ready_train, word_to_int)
        map_sentences_to_cat(X_test, self.ready_tagged_test, self.ready_test, word_to_int)

        assert(len(X_train[0]) == len(X_train[1]) == len(Y_train) == TRAIN_OPINIONS)
        assert (len(X_test[0]) == len(X_test[1]) == len(Y_test) == TEST_OPINIONS)

        weights = []
        for i in range(len(int_to_word)):
            try:
                weights.append(embbedings.word_to_emb[int_to_word[i]])
            except Exception:
                weights.append(embbedings.default_emb)
        return X_train, Y_train, X_test, Y_test, np.array(weights)

    def get_x_train_test_syntax_polarity(self, komn, pad=False):
        ''' For each word returns the sum of all dependencies'''
        x_train, x_test = [], []
        for i in range(len(self.ready_tagged)):
            sc = komn.get_syntactic_concatenation(self.ready_tagged[i])
            if pad:
                sc = pad_array(sc, 80)
            if i < TRAIN_SENTENCES:
                for o in list(self.ready_train.values())[i]['opinions']:
                   x_train.append(sc)
            else:
                for o in list(self.ready_test.values())[i-TRAIN_SENTENCES]['opinions']:
                    x_test.append(sc)
        return np.array(x_train), np.array(x_test)

    def get_x_train_test_syntax_polarity_sow(self, komn):
        ''' return len 600 sow '''
        x_train, x_test = self.get_x_train_test_syntax_polarity(komn, pad=False)
        x_train = [np.array(sum(e)) for e in x_train]
        x_test = [np.array(sum(e)) for e in x_test]
        return np.array(x_train), np.array(x_test)

    def get_x_train_test_polarity_sow(self, komn):
        x_train, x_test = self.get_x_train_test_syntax_polarity_sow(komn)
        x_train = x_train[:, 0:300]
        x_test = x_test[:, 0:300]
        return np.array(x_train), np.array(x_test)

    def get_normal_sentences_sow(self, embs, no_punct=False, no_stop=False):
        x_train, x_test = [], []
        for i in range(len(self.ready_tagged)):
            s = self.ready_tagged[i][0]
            if no_stop:
                s = self.text_preprocessor.remove_stopwords(s)
            if no_punct:
                s = self.text_preprocessor.remove_punctuation(s)
            if i < TRAIN_SENTENCES:
                for o in list(self.ready_train.values())[i]['opinions']:
                    sow = embs.get_SOW(s)
                    x_train.append(sow)
            else:
                for o in list(self.ready_test.values())[i - TRAIN_SENTENCES]['opinions']:
                    sow = embs.get_SOW(s)
                    x_test.append(sow)
        return np.array(x_train), np.array(x_test)

    def get_aspects_train_test(self, embs):
        X_train, _, X_test, _, _ = self.get_data_as_integers_and_emb_weights_polarity(embs)
        return np.array(X_train[1]), np.array(X_test[1])
