import pandas as pd
import numpy as np
from gensim.utils import unpickle
from utils import dump_gzip, load_gzip, preprocess

def make_word_to_emb_dictionary():
    embs = np.load("400dm_by_5lac_yelp.model.syn0.npy")
    embs_path = "400dm_by_5lac_yelp.model"
    obj = unpickle(embs_path)
    dexes = obj.index2word
    assert (len(embs) == len(dexes))
    words_to_emb = {}
    for i in range(0, len(dexes)):
        words_to_emb[dexes[i]] = embs[i]
    dump_gzip(words_to_emb, "words_to_emb")

def multiHotVectors(categories , name):
    #http://stackoverflow.com/questions/18889588/create-dummies-from-column-with-multiple-values-in-pandas
    dummies = pd.get_dummies(categories['category'])
    atom_col = [c for c in dummies.columns if '*' not in c]
    for col in atom_col:
        categories[col] = dummies[[c for c in dummies.columns if col in c]].sum(axis=1)

    categories.to_csv(name+".csv")
    return categories

def multiCategoryVectors(dataset, classes=12):
    labels = np.zeros((dataset.shape[0], classes))  # matrix of zeroes for accross all example labels
    for i in range(dataset.shape[0]):
        vec = dataset.values[i][2:14]
        labels[i] = vec
    print("Hot encoded vectors shape: ", labels.shape)
    return labels

def get_SOW(sentence, embeddings):
    # beacuse we are summing: if word is not in embeddings we just append a vector of zeroes
    sentence = [embeddings[w] if w in embeddings.keys()
                else np.zeros((1, len(embeddings["tree"]))) for w in sentence]
    running_sum = np.zeros((1, len(sentence[0])))
    for emb in sentence:
        running_sum = np.add(running_sum, emb)
    return running_sum

def get_data():
    training_output_File = 'train.txt'
    gold_output_File = 'test.txt'

    train = pd.read_csv(training_output_File, header=None, delimiter="\t", quoting=3, names=['review', 'category'])
    test = pd.read_csv(gold_output_File, header=None, delimiter="\t", quoting=3, names=['review', 'category'])
    train = multiHotVectors(train, 'test_labels')
    test = multiHotVectors(test, 'train_labels')

    x_train = train["review"]
    y_train = np.array(multiCategoryVectors(train, classes=12)).astype(int)
    x_test = test["review"]
    y_test = np.array(multiCategoryVectors(test, classes=12)).astype(int)

    embs = load_gzip("words_to_emb")

    x_train = [preprocess(s, return_list=True) for s in x_train]
    x_train = [get_SOW(x, embs) for x in x_train]

    x_test = [preprocess(s, return_list=True) for s in x_test]
    x_test = [get_SOW(x, embs) for x in x_test]
    return np.squeeze(np.asarray(x_train[:])), y_train, \
           np.squeeze(np.asarray(x_test[:])), y_test