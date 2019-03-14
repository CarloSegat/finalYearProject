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
labels3 = {'positive': 2,'neutral': 1,'negative': 0,'2' : 2,'1' : 2,'0' : 1,'-1': 0,'-2': 0}
labels5 = {'2' : 4,'1' : 3,'0' : 2,'-1': 1,'-2': 0,'positive': 3,'neutral': 2,'negative': 1}

text_processor = TextPreProcessor(
	normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
		'time', 'url', 'date', 'number'],
	unpack_contractions=True,
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

def preprocess(sentence, remove_stop=False, return_list=False):
	decontractedSentence = decontracted(sentence)
	preProcessed = text_processor.pre_process_doc(decontractedSentence)
	if remove_stop:
		stops = set(stopwords.words("english"))
		preProcessed = [word for word in preProcessed if word not in stops]
	if return_list:
		return np.array(preProcessed)
	else:
		return " ".join(preProcessed)

def makeTokenizer(sentences, wordsUsed):
	tokenizer = Tokenizer(wordsUsed)
	tokenizer.fit_on_texts(list(sentences))
	return tokenizer

def decontracted(phrase):
	phrase = re.sub(r"won\'t", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"he\'s", "he is", phrase)
	phrase = re.sub(r"she\'s", "she is", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase

def classification_report(y_true, y_pred, labels):
    '''Taken from kers-contrib
    https://github.com/keras-team/keras-contrib/blob/master/examples/conll2000_chunking_crf.py'''
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    'recall',
                                                    'precision',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')

def load_Suresh_embeddings(int_to_word, embedding_length=300):
	output_file = ROOT_DIR + '/data/suresh_loaded.pkl.gz'
	output_file2 = ROOT_DIR + '/data/suresh_loaded_all.pkl.gz'
	embeddingsPath = ROOT_DIR + '/data/wiki_extvec.gz'

	#if os.path.exists(output_file):
	#	return load_gzip(output_file)[]
	fEmbeddings = gzip.open(embeddingsPath, "r")
	all_word_to_emb = create_word_to_embedding_dictionary(embedding_length, fEmbeddings)
	dump_gzip("all_embeddings", all_word_to_emb, output_file2)
	exit() # TODO TEMPORARY
	embeddings = map_vocabulary_to_embeddings(all_word_to_emb, int_to_word)
	embeddings = np.array(embeddings)
	dump_gzip({"embeddings": embeddings}, output_file)
	return embeddings

def create_word_to_embedding_dictionary(embedding_length, raw_embeddings):
	all_word_to_emb = {}
	for line in raw_embeddings:
		split = line.decode('utf-8').strip().split(" ")
		word = split[0]
		assert (len(split[1:]) == embedding_length)
		vector = np.array([float(num) for num in split[1:]])
		all_word_to_emb[word] = vector
	return all_word_to_emb

def dump_gzip(data, output_file: str):
	f = gzip.open(output_file, 'wb')
	pkl.dump(data, f)
	f.close()

def load_gzip(file):
	file = gzip.open(file, "r")
	data = pkl.load(file)
	file.close()
	return data

def map_vocabulary_to_embeddings(word_to_embedding_dictionary, vocabulary):
	embedding_length = len(word_to_embedding_dictionary['i'])
	default_embedding = np.random.uniform(-0.25, 0.25, embedding_length)
	words_that_have_embeddings = word_to_embedding_dictionary.keys()
	embeddings = []
	for i in range(0, len(vocabulary)):
		current_vocabulary_word = vocabulary[i]
		if current_vocabulary_word in words_that_have_embeddings:
			embeddings.append(word_to_embedding_dictionary[current_vocabulary_word])
		else:
			embeddings.append(default_embedding)
	return embeddings

def save_embeddings(model_final):
	'''
	Based on the fact that embedding layer is the second after input
	'''
	weights = model_final.layers[1].get_weights()[0]
	data = {'tuned_embeddings': weights}
	f = gzip.open("tuned_embeddings", 'wb')
	pkl.dump(data, f)
	f.close()

def make_dictionary_of_komn(embeddingsPath, output_file):
	'''
	Generate dictionary mapping words to int; will be used to tokenize amazon
	reviews when fine tuning embeddings
	'''

	embedding = gzip.open(embeddingsPath, "r")
	word_to_int = {}
	c = 0
	for line in embedding:
		split = line.decode('utf-8').strip().split(" ")
		word = split[0]
		word_to_int[word] = c
		c += 1

	data = {'word_to_int': word_to_int, 'length': c -1}
	dump_gzip(data, output_file)

class Embedizer():

	def __init__(self, embedding_path):
		self.embeddings = load_gzip(embedding_path)


def loadData(files, elementsPerLine, maxLen=90, turnInputIntoEmbedding=False, cleaning=False, classes=3):
	x, y, xt, yt = [], [], [], []
	for file in files["train"]:
		temp_x, temp_y = loadRaw(file[0], file[1], classes)
		x = x + temp_x
		y = y + temp_y
	for file in files["test"]:
		temp_xt, temp_yt = loadRaw(file[0], file[1], classes)
		xt = xt + temp_xt
		yt = yt + temp_yt
	if cleaning:
		x = [preprocess(sentence) for sentence in x]
		xt = [preprocess(sentence) for sentence in xt]
		x = [s[1:-2].strip() if (s[0] == s[-1] == "\"") else s.strip() for s in x]
		xt = [s[1:-2].strip() if (s[0] == s[-1] == "\"") else s.strip() for s in xt]
	# create tokenizer and threfore word index using both sentences fro test and train
	tokenizer = makeTokenizer(x + xt, 20000)
	x = tokenizer.texts_to_sequences(x)
	x = pad_sequences(x, maxLen, padding='post')

	xt = tokenizer.texts_to_sequences(xt)
	xt = pad_sequences(xt, maxLen, padding='post')
	int_to_word = {int: word for word, int in tokenizer.word_index.items()}
	int_to_word[0] = "PAD"

	embeddings = load_Suresh_embeddings(int_to_word)

	return x, y, xt, yt, int_to_word, embeddings


def loadRaw(fileName, elementsPerLine, classes, ):
	x = []
	y = []
	with open(fileName, 'r') as fileName:
		lines = fileName.read().splitlines()
		for line in lines:
			idClassSetence = re.split(r'\t', line.strip())
			try:
				# sometimes a tweet is not just one element of a list but it gets split in two
				if len(idClassSetence) == elementsPerLine:
					sentenceIndex = -1
					classIndex = -2
				else:
					sentenceIndex = -2
					classIndex = -3
				if (idClassSetence[sentenceIndex].strip().lower() == "not available"):
					continue
				x.append(idClassSetence[sentenceIndex])
				y.append(idClassSetence[classIndex].strip())
			except Exception:
				print("end of file")
	if classes == 3:
		y = [labels3[rawClass] for rawClass in y]
	elif classes == 5:
		y = [labels5[rawClass] for rawClass in y]
	else:
		raise RuntimeError
	return x, y
