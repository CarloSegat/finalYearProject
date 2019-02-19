from pprint import pprint  # pretty-printer
from collections import defaultdict
import re
from gensim import *
from gensim.test.utils import get_tmpfile
from gensim.test.utils import common_texts
from gensim.models import *
import pdb

model = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)  
vector = model.wv['computer']
pdb.set_trace()
documents = ["Human machine interface for lab abc computer applications",
              "A survey of user opinion of computer system response time",
              "The EPS user interface management system",
              "System and human system engineering testing of EPS",
              "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
              "The intersection graph of paths in trees",
              "Graph minors IV Widths of trees and well quasi ordering",
              "Graph minors A survey"]
list_sentences_train = []
y = []
with open('2017ATrain.txt', 'r') as input:
	lines = input.read().splitlines()
	
	for line in lines:
		idClassSetence = re.split(r'\t', line.strip())
		
		try:
			list_sentences_train.append(idClassSetence[2])
			y.append(idClassSetence[1])
		except Exception:
			print("end of dile")
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]for document in list_sentences_train]

path = get_tmpfile("word2vec.model")
model = Word2Vec(list_sentences_train, size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

frequency = defaultdict(int)

for text in texts:
	for token in text:
		frequency[token] += 1


texts = [[token for token in text if frequency[token] > 1]for text in texts]
dictionary = corpora.Dictionary(texts)
tmp_fname = get_tmpfile("dictionary")
dictionary.save_as_text("myDictionaryy")
#dictionary.save('/myDict.dict')  # store the dictionary, for future reference