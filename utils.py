import re
import numpy as np
import pdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.classes.spellcorrect import SpellCorrector
from nltk.corpus import stopwords
import nltk
from collections import Counter
import pickle as pkl
import gzip
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))



def make_tokenizer(sentences, wordsUsed):
    tokenizer = Tokenizer(wordsUsed)
    tokenizer.fit_on_texts(list(sentences))
    return tokenizer


def dump_gzip(data, output_file: str):
    f = gzip.open(output_file, 'wb')
    pkl.dump(data, f)
    f.close()

def load_gzip(file):
    file = gzip.open(file, "r")
    data = pkl.load(file)
    file.close()
    return data