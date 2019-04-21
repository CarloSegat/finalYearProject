import re
import string

import numpy as np
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
import nltk
from nltk.corpus import stopwords

class TextPreprocessor():

    def __init__(self):

        self.text_processor_options = TextPreProcessor(
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'url', 'date', 'number'],
            unpack_contractions=False,
            annotate={"allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens
            # corpus from which the word statistics are going to be used
            # for word segmentation and correction
            segmenter="english",
            corrector="english",
            unpack_hashtags=False,  # perform word segmentation on hashtags
            spell_correct_elong=False,  # spell correction for elongated words
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,
            # list of dictionaries, for replacing tokens extracted from the text,
            dicts=[emoticons]
        )

    def do_ekphrasis_preprocessing(self, sentences):
        if isinstance(sentences, str):
            return self.text_processor_options.pre_process_doc(sentences)

        assert (type(sentences).__module__ == np.__name__)
        preprocessed = [self.text_processor_options.pre_process_doc(s) for s in sentences]
        return np.array(preprocessed)

    def do_decontraction(self, sentences):
        if isinstance(sentences, str):
            sentences = np.array([sentences])
        assert(type(sentences).__module__ == np.__name__)
        preprocessed = []
        for s in sentences:
            ''' Does not deal with 'd as it is ambiguous'''
            s = re.sub(r"[W, w]on\'t", "will not", s)
            s = re.sub(r"[C, c]an\'t", "can not", s)
            s = re.sub(r"[C, c]annot", "can not", s)
            s = re.sub(r"n\'t", " not", s)
            s = re.sub(r"\'re", " are", s)
            s = re.sub(r"[H, h]e\'s", "he is", s)
            s = re.sub(r"[S, s]he\'s", "she is", s)
            s = re.sub(r"[I, i]t\'s", "it is", s)
            s = re.sub(r"\'ll", " will", s)
            s = re.sub(r"\'ve", " have", s)
            s = re.sub(r"\'m", " am", s)
            s = re.sub(r"[D, d]idn\'t", "did not", s)
            preprocessed.append(s)
        return np.array(preprocessed)

    def remove_stopwords(self, sentence):
        assert (type(sentence).__module__ == np.__name__ or isinstance(sentence, list))
        new_sentence = []
        for w in sentence:
            if not self.is_word_stop(w):
                new_sentence.append(w)
        return np.array(new_sentence)

    def remove_punctuation(self, sentence):
        assert (type(sentence).__module__ == np.__name__ or isinstance(sentence, list))
        new_sentence = []
        for w in sentence:
            if not self.is_word_punctuation(w):
                new_sentence.append(w)
        return np.array(new_sentence)

    def is_word_stop(self, w):
        return w in set(stopwords.words('english'))

    def is_word_punctuation(self, w):
        return w.translate(str.maketrans('', '', string.punctuation)) == ''

if __name__ == '__main__':
    tp = TextPreprocessor()
    p = tp.do_decontraction(["I'm waiting you, can't you see she's here"])
    print(p)