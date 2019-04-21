# From: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs

import numpy as np
import random
from keras.preprocessing.sequence import pad_sequences


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    sentences = []
    sentence = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split(' ')
        sentence.append([splits[0], splits[-1]])

    if len(sentence) > 0:
        sentences.append(sentence)
        sentence = []
    return sentences


# define casing s.t. NN can use case information to learn patterns
def getCasing(word, caseLookup):
    casing = 'other'

    numDigits = 0
    for char in word:
        if char.isdigit():
            numDigits += 1

    digitFraction = numDigits / float(len(word))

    if word.isdigit():  # Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # All lower case
        casing = 'allLower'
    elif word.isupper():  # All upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'

    return caseLookup[casing]


# return batches ordered by words in sentence
def createEqualBatches(data):
    
    
    # num_words = []
    # for i in data:
    #     num_words.append(len(i[0]))
    # num_words = set(num_words)
    
    n_batches = 100
    batch_size = len(data) // n_batches
    num_words = [batch_size*(i+1) for i in range(0, n_batches)]
    
    batches = []
    batch_len = []
    z = 0
    start = 0
    for end in num_words:
        # print("start", start)
        for batch in data[start:end]:
            # if len(batch[0]) == i:  # if sentence has i words
            batches.append(batch)
            z += 1
        batch_len.append(z)
        start = end

    return batches, batch_len

def createBatches(data):
    lengths_sentences = []
    for i in data:
        lengths_sentences.append(len(i[0]))
    lengths_sentences = set(lengths_sentences)
    batches = []
    batch_len = []
    z = 0
    for i in lengths_sentences:
        for batch in data:
            if len(batch[0]) == i:
                if batch[4][i+1, 0] != 0.0:
                    print(batch[4][i+1, 0])
                batch = [batch[0], batch[1], batch[2], batch[3], batch[4][0:i, :]]
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len


# Converts sentences into lists of:
# word indices, case indices, character indices, label indices
def createMatrices(sentences, word2Idx, label2Idx, case2Idx, char2Idx):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence in sentences:
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []

        for word, char, label in sentence:
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                try:
                    charIdx.append(char2Idx[x])
                except Exception:
                    print(x)
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices])

    return dataset

def createMatrices_syntax(sentences, syntax_sentences, word2Idx, label2Idx,
                          case2Idx, char2Idx, no_stop, no_punct, tp):
    assert (len(sentences) == len(syntax_sentences))
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']

    dataset = []

    wordCount = 0
    unknownWordCount = 0

    for sentence, syntax_sentence in zip(sentences, syntax_sentences):
        wordIndices = []
        caseIndices = []
        charIndices = []
        labelIndices = []


        for word, char, label in sentence:
            if no_stop:
                if tp.is_word_stop(word):
                    continue
            if no_punct:
                if tp.is_word_punctuation(word):
                    continue
            wordCount += 1
            if word in word2Idx:
                wordIdx = word2Idx[word]
            elif word.lower() in word2Idx:
                wordIdx = word2Idx[word.lower()]
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                try:
                    charIdx.append(char2Idx[x])
                except Exception:
                    print(x)
            # Get the label and map to int
            wordIndices.append(wordIdx)
            caseIndices.append(getCasing(word, case2Idx))
            charIndices.append(charIdx)
            labelIndices.append(label2Idx[label])

        dataset.append([wordIndices, caseIndices, charIndices, labelIndices, syntax_sentence])

    return dataset

def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
        
        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), np.asarray(char)

def iterate_minibatches_syntax(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
        caseing = []
        char = []
        labels = []
        syntax = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t, c, ch, l, synt = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            caseing.append(c)
            char.append(ch)
            labels.append(l)
            syntax.append(synt)

        yield np.asarray(labels), np.asarray(tokens), np.asarray(caseing), \
              np.asarray(char), np.asarray(syntax)


        # returns data with character information in format
# [['EU', ['E', 'U'], 'B-ORG\n'], ...]
def addCharInformation(Sentences):
    for i, sentence in enumerate(Sentences):
        for j, data in enumerate(sentence):
            chars = [c for c in data[0]]
            Sentences[i][j] = [data[0], chars, data[1]]
    return Sentences


# 0-pads all words
def padding(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[2]
        for x in char:
            maxlen = max(maxlen, len(x))
    for i, sentence in enumerate(Sentences):
        Sentences[i][2] = pad_sequences(Sentences[i][2], 52, padding='post')
    return Sentences
