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

labels3 = {
		'positive': 1, 
		'neutral': 0, 
		'negative': -1,
		'2' : 1, 
		'1' : 1, 
		'0' : 0, 
		'-1': -1,
		'-2': -1}

labels5 = {
		'2' : 2, 
		'1' : 1, 
		'0' : 0,
		'-1': -1, 
		'-2': -2,
		'positive': 1, 
		'neutral': 0, 
		'negative': -1}
		


sp = SpellCorrector(corpus="english") 
seg_eng = Segmenter(corpus="english") 		
text_processor = TextPreProcessor(
	normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
		'time', 'url', 'date', 'number'],

	annotate={"hashtag", "allcaps", "elongated", "repeated",
		'emphasis', 'censored'},
	fix_html=True,  # fix HTML tokens
	
	# corpus from which the word statistics are going to be used 
	# for word segmentation and correction
	segmenter="twitter", 
	corrector="twitter", 
	
	unpack_hashtags=True,  # perform word segmentation on hashtags
	unpack_contractions=True,  # Unpack contractions (can't -> can not)
	spell_correct_elong=False,  # spell correction for elongated words
	
	# select a tokenizer. You can use SocialTokenizer, or pass your own
	# the tokenizer, should take as input a string and return a list of tokens
	tokenizer=SocialTokenizer(lowercase=True).tokenize,
	
	# list of dictionaries, for replacing tokens extracted from the text,
	# with other expressions. You can pass more than one dictionaries.
	dicts=[emoticons]
)

def loadData(files, elementsPerLine, maxLen=90, turnInputIntoEmbedding=False, cleaning=False, classes=3):
	x, y = loadRaw(files["train"][0], files["train"][1], classes)
	xt, yt = loadRaw(files["test"][0], files["test"][1], classes)
	if cleaning:
		x = [preprocess(sentence) for sentence in x]
		xt = [preprocess(sentence) for sentence in xt]
		x = [s[1:-2].strip() if (s[0] == s[-1] == "\"") else s.strip() for s in x] 
		xt = [s[1:-2].strip() if (s[0] == s[-1] == "\"") else s.strip() for s in xt] 	
		
	tokenizer = makeTokenizer(x + xt, 20000)
	x = tokenizer.texts_to_sequences(x)
	x = pad_sequences(x, maxLen, padding='post')
	
	xt = tokenizer.texts_to_sequences(xt)
	xt = pad_sequences(xt, maxLen, padding='post')
	intToWord = {int: word for word, int in tokenizer.word_index.items()}
	intToWord[0] = "PAD"
	#embedding = voc2vec(300, wordsUsed)
	#if turnInputIntoEmbedding:
		# all words of the index are in the embedding, after tokenisation only words of the index are left in the senences
		#x = [[embedding[word] for word in sentence] for sentence in x]
	return x, y, xt, yt, intToWord

def loadRaw(fileName, elementsPerLine, classes,):
	''' Dont need to return a numpy list'''
	
	x = []
	y = []
	with open(fileName, 'r') as input:
		lines = input.read().splitlines()
		for line in lines:
			
				
			idClassSetence = re.split(r'\t', line.strip())
			try:
				if len(idClassSetence) == elementsPerLine:
					sentenceIndex = -1
					classIndex = -2
				else:
					sentenceIndex = -2
					classIndex = -3
				if(idClassSetence[sentenceIndex].strip().lower() == "not available"):
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
	
def preprocess(sentence):
	decontractedSentence = decontracted(sentence)
	preProcessed = text_processor.pre_process_doc(decontractedSentence)
	#stops = set(stopwords.words("english"))
	#filtered_words = [word for word in preProcessed if word not in stops]
	return " ".join(preProcessed)

def makeTokenizer(sentences, wordsUsed):
	tokenizer = Tokenizer(wordsUsed)
	tokenizer.fit_on_texts(list(sentences))
	return tokenizer

def voc2vec(embedLength, wordsUsed):
	'''
	Generate the embedding matrix from the words in the index
	'''
	googleModel = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)  
	wv = googleModel.wv
	# TODO mean and deviation of googles data
	embedding_matrix = np.random.normal(0, 3, (wordsUsed, embedLength)) 
	for word, index in wordIndex.items(): # list of tuples
		if index >= wordsUsed:
			continue # Only use the most frequent words up to wordsUsed
		try:
			embedding_vector = wv[word]
			embedding_matrix[index] = embedding_vector
		except KeyError:
			embedding_matrix[index] = embedding_matrix[index]
	return embedding_matrix

def decontracted(phrase):
	phrase = re.sub(r"won\'t", "will not", phrase)
	phrase = re.sub(r"can\'t", "can not", phrase)
	phrase = re.sub(r"n\'t", " not", phrase)
	phrase = re.sub(r"\'re", " are", phrase)
	phrase = re.sub(r"he\'s", "he is", phrase)
	phrase = re.sub(r"she\'s", "she is", phrase)
	phrase = re.sub(r"\'d", " would", phrase)
	phrase = re.sub(r"\'ll", " will", phrase)
	phrase = re.sub(r"\'ve", " have", phrase)
	phrase = re.sub(r"\'m", " am", phrase)
	return phrase
