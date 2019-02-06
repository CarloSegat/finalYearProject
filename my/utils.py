import re
import numpy as np
import pdb

def loadSentences(fileName):
	''' Dont need to return a numpy list'''
	list_sentences_train = []
	with open(fileName, 'r') as input:
		lines = input.read().splitlines()
		for line in lines:
			idClassSetence = re.split(r'\t', line.strip())
			try:
				list_sentences_train.append(idClassSetence[2])
			except Exception:
				print("end of file")
	return list_sentences_train
	
def loadTargets(fileName):
	''' For each sentence wanna output a 3 elements tuple where the element corresponding to the 
	real class is 1'''
	labels = {
		'positive': np.array([1, 0, 0]), 
		'neutral': np.array([0, 1, 0]), 
		'negative': np.array([0, 0, 1])
	}
	y = np.empty((0,3), int)
	with open(fileName, 'r') as input:
		lines = input.read().splitlines()
		for line in lines:
			idClassSetence = re.split(r'\t', line.strip())
			try:
				y = np.vstack((y, labels[idClassSetence[1]]))
			except Exception:
				print("end of file")
	return y #np.c_[y]