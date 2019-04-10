import gzip
import re
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from gensim.utils import unpickle
from utils import ROOT_DIR, load_gzip, dump_gzip
from pathlib import Path
import numpy as np

class Embedding():

    def __init__(self, vocabulary):
        self.base_path = ROOT_DIR + "\embeddings\\"
        self.int_to_word = None
        self.word_to_emb = None
        self.vocabulary = vocabulary

    def get_wordIndex_and_embeddingLayer(self, vocabulary):
        '''EMbeding layer in Keras works by substituting integers with an array
        at corresponding index.
        Therefore the order of the embeddigns need to match the integers used to
        encode words.'''

        ordered_embeddings = []
        ordered_word_to_int = {}

        for word in vocabulary:
            try:
                embedding = self.word_to_emb[word]
                ordered_word_to_int[word] = len(ordered_embeddings)
                ordered_embeddings.append(embedding)
            except Exception:
                pass

        return ordered_word_to_int, ordered_embeddings

    def get_SOW(self, sentence):
        if not isinstance(sentence, list):
            if(isinstance(sentence, str)):
                sentence = sentence.split(" ")
        running_sum = np.zeros(len(self.word_to_emb['price']))
        for word in sentence:
            try:
                running_sum = np.add(running_sum, self.word_to_emb[word])
            except Exception:
                pass # When we don't know the embedding for the word
        return running_sum

    def get_word_emb_list(self, sentence):
        if not isinstance(sentence, list):
            if(isinstance(sentence, str)):
                sentence = sentence.split(" ")
        embs = []
        for word in sentence:
            try:
                embs.append(self.word_to_emb[word])
            except Exception:
                pass # When we don't know the embedding for the word
        return embs

    def get_word_to_emb(self):
        return self.word_to_emb

    def get_int_to_word(self):
        return self.int_to_word

class Glove(Embedding):

    def __init__(self, size, vocabulary):
        super(Glove, self).__init__(vocabulary)
        self.size = size
        self.base_path = self.base_path + 'glove\\'
        self.path_raw = self.base_path + 'glove.6B.' + str(size) + 'd.txt'
        if Path(self.base_path + 'Glove'+ str(self.size)+'_word_to_emb').is_file():
            self.word_to_emb = load_gzip(self.base_path + 'Glove'+ str(self.size)+'_word_to_emb')
        else:
            self.load()

    def load(self):
        word_to_emb = {}
        embeddings = open(self.path_raw, "r", encoding='latin-1')
        for e in embeddings:
            e = e.strip().split(" ")
            word = e[0]
            vector = e[1::]
            if word in self.vocabulary:
                word_to_emb[word] = vector
                if len(word_to_emb) == len(self.vocabulary):
                    # got all words in vocabulary
                    break
        dump_gzip(word_to_emb, self.base_path + 'Glove' + str(self.size) + '_word_to_emb')

class Komn(Embedding):

    def __init__(self, vocabulary):
        super(Komn, self).__init__(vocabulary)
        self.base_path = self.base_path + 'komn\\'
        self.path_raw = self.base_path + 'wiki_extvec.gz'
        if Path(self.base_path + 'Komn_word_to_emb').is_file():
            self.word_to_emb = load_gzip(self.base_path + 'Komn_word_to_emb')
        else:
            self.load_simple_words()

    def load_simple_words(self):

        def keep_normal_words_vectors(vector, word, word_to_emb):
            if len(lowercase.findall(word)) > 0:
                pass
            else:
                word_to_emb[word] = vector

        def get_word_and_vector(e):
            split = e.decode('latin-1').strip().split(" ")
            word = split[0]
            assert (len(split[1:]) == 300)
            vector = np.array([float(num) for num in split[1:]])
            return vector, word

        word_to_emb = {}
        embeddings = gzip.open(self.path_raw, "r")
        lowercase = re.compile('.*_.*')
        for e in embeddings:
            vector, word = get_word_and_vector(e)
            if word in self.vocabulary:
                keep_normal_words_vectors(vector, word, word_to_emb)
                if len(word_to_emb) == len(self.vocabulary):
                    # got all words in vocabulary
                    break
        dump_gzip(word_to_emb, self.base_path + 'Komn_word_to_emb')



class Google(Embedding):
    ''' ONLY EMBEDDINGS FOR TRAIN+TEST VOCABULARY
    (Since 3M embeddings are too much for my memory to deal with)'''

    def __init__(self, vocabulary):
        super(Google, self).__init__(vocabulary)
        self.base_path = self.base_path + 'google\\'
        self.path_raw = self.base_path + 'GoogleNews-vectors-negative300.bin'
        if Path(self.base_path + 'Google_word_to_emb').is_file():
            self.word_to_emb = load_gzip(self.base_path + 'Google_word_to_emb')
        else:
            self.load()

    def load(self):
        wv_from_bin = KeyedVectors.load_word2vec_format(self.path_raw, binary=True)
        int_to_word = wv_from_bin.index2word
        word_to_int = {v: k for k, v in enumerate(int_to_word)}
        int_to_embedding = {}
        word_to_embedding = {}
        for w in self.vocabulary:
            try:
                int_to_embedding[word_to_int[w]] = wv_from_bin.vectors[word_to_int[w]]
                word_to_embedding[w] = wv_from_bin.vectors[word_to_int[w]]
            except Exception:
                pass
        dump_gzip(word_to_int, self.base_path + "Google_word_to_int")
        dump_gzip(int_to_word, self.base_path + "Google_int_to_word")
        dump_gzip(int_to_embedding, self.base_path + "Google_int_to_emb")
        dump_gzip(word_to_embedding, self.base_path + "Google_word_to_emb")

        self.word_to_emb = load_gzip(self.base_path + 'Google_word_to_emb')


class Yelp(Embedding):

    def __init__(self, vocabulary):
        super(Yelp, self).__init__(vocabulary)
        self.base_path = self.base_path + 'yelp\\'
        self.file_name = '400dm_by_5lac_yelp.model'
        if Path(self.base_path + "Yelp_word_to_emb").is_file():
            self.word_to_emb = load_gzip(self.base_path + "Yelp_word_to_emb")
        else:
            self.load()

    def load(self):
        '''Simply using gensim Word2Vec.load throws an error.
        This function is a work around.'''
        try:
            embs = np.load(self.base_path +"400dm_by_5lac_yelp.model.syn0.npy")
            embs_path = self.base_path + "400dm_by_5lac_yelp.model"
            obj = unpickle(embs_path)
            dexes = obj.index2word
            assert (len(embs) == len(dexes))
            words_to_emb = {}
            for i in range(0, len(dexes)):
               words_to_emb[dexes[i]] = embs[i]
            dump_gzip(words_to_emb, self.base_path + "Yelp_word_to_emb")
            self.word_to_emb = load_gzip(self.base_path + "Yelp_word_to_emb")
        except Exception:
            print("Missing Yelp word embeddings. Download should be available",
                  "at this git hub repo: Nomiluks/Aspect-Category-Detection-Model")

# def save_embeddings(model_final):
#     '''
# 	Based on the fact that embedding layer is the second after input
# 	'''
#     weights = model_final.layers[1].get_weights()[0]
#     data = {'tuned_embeddings': weights}
#     f = gzip.open("tuned_embeddings", 'wb')
#     #pkl.dump(data, f)
#     f.close()
#




if __name__ == '__main__':
    from SemEval import SemEvalData

    sem = SemEvalData()
    vocabulary = sem.make_vocabulary()
    k = Komn(vocabulary)
    k. load_simple_words()

    y = Yelp(vocabulary)
    a = y.word_to_emb['tree']
    g = Glove(100)
    g = Glove(300)

    e, r = g.get_wordIndex_and_embeddingLayer(vocabulary)
    p = 3