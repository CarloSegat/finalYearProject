import re
import numpy as np
import pdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

wordIndex

labels3 = {
		'positive': [1, 0, 0], 
		'neutral': [0, 1, 0], 
		'negative': [0, 0, 1]
}

labels5 = {
		'2' : [1, 0, 0, 0, 0], 
		'1' : [0, 1, 0, 0, 0], 
		'0' : [0, 0, 1, 0, 0],
		'-1': [0, 0, 0, 1, 0], 
		'-2': [0, 0, 0, 0, 1]}

def loadData(path, wordsUsed, maxLen=90, applyEmbedding=False, cleaning=False):
	x, y = loadRaw(path)
	if cleaning:
		x = map(clean, x)
	x = tokenize(x, wordsUsed)
	x = pad_sequences(x, maxLen, padding='post')
	embedding = voc2vec(300, wordsUsed)
	if applyEmbedding:
		# all words of the index are in the embedding, after tokenisation only words of the index are left in the senences
		x = [[embedding[word] for word in sentence] for sentence in x]
	pdb.set_trace()
	return x, y, embedding

def loadRaw(fileName):
	''' Dont need to return a numpy list'''
	x = []
	y = []
	with open(fileName, 'r') as input:
		lines = input.read().splitlines()
		for line in lines:
			idClassSetence = re.split(r'\t', line.strip())
			try:
				if(idClassSetence[-1].strip() == "Not Available"):
					continue
				x.append(idClassSetence[-1])
				y.append(idClassSetence[-2].strip())
			except Exception:
				print("end of file")
	if len(set(y)) == 3:
		y = [labels3[yy] for yy in y]
	elif len(set(y)) == 5:
		y = [labels5[yy] for yy in y]
	return x, y
	
def clean(sentence):
	return setence.strip()
	
def tokenize(sentences, wordsUsed):
	''' sentences are tokenized into ints. 
	Rare words will be excluded'''
	tokenizer = Tokenizer(wordsUsed)
	tokenizer.fit_on_texts(list(sentences))
	xTokenized= tokenizer.texts_to_sequences(sentences)
	global wordIndex 
	wordIndex = tokenizer.word_index
	return xTokenized

def voc2vec(embedLength, wordsUsed):
	'''
	Generate the embedding matrix from the words in the index
	'''
	googleModel = KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)  
	wv = googleModel.wv
	# TODO mean and deviation of googles data
	embedding_matrix = np.random.normal(0, 3, (wordsUsed, embedLength)) 
	for word, index in wordIndex.items(): # list of tuples
		if index >= wordsUsed:
			continue # Only use the most frequent words up to wordsUsed
		try:
			embedding_vector = wv[word]
			embedding_matrix[wordIndex] = embedding_vector
		except KeyError:
			embedding_matrix[wordIndex] = embedding_matrix[wordIndex]
	return embedding_matrix
	
five =  "..\\data\\SEData\\2017\\4c-english\\4C-English\\SemEval2017-task4-dev.subtask-CE.english.INPUT.txt"	
three = '..\\data\\SEData\\2017\\4a-english\\4A-English\\SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
loadData(five, 20000)