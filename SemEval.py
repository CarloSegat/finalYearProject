import os
import re
import xml
from pathlib import Path
from keras.utils import to_categorical
from django.utils.encoding import smart_str
import pandas as pd
import numpy as np
from gensim.utils import unpickle

from Embeddings import Komn
from TextPreprocessor import TextPreprocessor
from utils import dump_gzip, load_gzip, ROOT_DIR, make_tokenizer

TRAIN_SENTENCES = 2000
TEST_SENTENCES = 676
TRAIN_OPINIONS = 2507
TEST_OPINIONS = 859

ASPECT_CATEGORIES = {'DRINKS#STYLE_OPTIONS': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     'LOCATION#GENERAL': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                     'AMBIENCE#GENERAL': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     'FOOD#PRICES': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     'FOOD#QUALITY': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                     'RESTAURANT#PRICES': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                     'DRINKS#QUALITY': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     'DRINKS#PRICES': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     'FOOD#STYLE_OPTIONS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                     'RESTAURANT#GENERAL': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                     'SERVICE#GENERAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                     'RESTAURANT#MISCELLANEOUS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                     }

class SemEvalData():
    '''SemEval2016_task5_subtask1'''

    def __init__(self):
        self.path_train_raw = ROOT_DIR + '/data/SemEval2016-Task5-ABSA/SB1/REST/ABSA16_Restaurants_Train_SB1_v2.xml'
        self.path_test_raw = ROOT_DIR + '/data/SemEval2016-Task5-ABSA/SB1/REST/EN_REST_SB1_TEST.xml.gold'
        self.train_file_name = 'SemEval2016_task5_subtask1_train_ready'
        self.test_file_name = 'SemEval2016_task5_subtask1_test_ready'
        self.path_train_ready = ROOT_DIR + '/data/' + self.train_file_name
        self.path_test_ready = ROOT_DIR + '/data/' + self.test_file_name

        self.text_preprocessor = TextPreprocessor()

        if Path(self.path_train_ready).is_file() and Path(self.path_test_ready).is_file():
            self.ready_train = load_gzip(self.path_train_ready)
            self.ready_test = load_gzip(self.path_test_ready)

        else:
            self.load_train_and_test()

    def make_vocabulary(self):
        ''' List of words used '''
        list_of_lists = self.get_all_sentences()
        all_words = [w for sentence in list_of_lists for w in sentence]
        return set(all_words)

    def get_all_sentences(self):
        train_sentences = self.get_train_sentences()
        test_sentences = self.get_test_sentences()
        return np.concatenate((train_sentences, test_sentences))

    def get_data_sow_and_oneHotVector(self, embedding):

        def assert_is_one_hot_vector(multiclass_label):
            temp = multiclass_label > 1
            assert (not temp.any())

        def make_multilabel_1hot_vector(aspect_categories):
            multiclass_label = np.zeros(12)
            for ac in aspect_categories:
                multiclass_label = np.add(multiclass_label, ASPECT_CATEGORIES[ac])
            assert_is_one_hot_vector(multiclass_label)
            return multiclass_label

        train_x_train_y_test_x_test_y = []
        for data in [self.ready_train, self.ready_test]:
            x, y = [], []
            for e in data.values():
                sentence = e['sentence']
                aspect_categories = []
                for opinion in e['opinions']:
                    aspect_categories.append(opinion['category'])
                # some sentences have more opinions of the same type.
                # Ignored and treated as one instance.
                aspect_categories = list(set(aspect_categories))
                x.append(embedding.get_SOW(sentence))
                y.append(make_multilabel_1hot_vector(aspect_categories))
            train_x_train_y_test_x_test_y.append(np.array(x))
            train_x_train_y_test_x_test_y.append(np.array(y, dtype=np.int32))
        return train_x_train_y_test_x_test_y



    def load_train_and_test(self):
        ''' Creates files of this format:
        {sentence_id: { 'sentence':[sentence], 'opinions':[{}, {}]}'''

        for file_in, file_out in [(self.path_train_raw, self.path_train_ready),
                                  (self.path_test_raw, self.path_test_ready)]:
            opinion_count = 0
            ready = {}
            reviews = xml.etree.ElementTree.parse(file_in).getroot()
            for review in reviews:
                for sentences in review:
                    for sentence in sentences:
                        ready[sentence.get("id")] = {}
                        this_id = sentence.get("id")
                        this_sentence = ''
                        for data in sentence:
                            if data.tag == "text":
                                this_sentence = data.text
                                this_sentence = self.text_preprocessor.do_decontraction(this_sentence)
                                this_sentence = self.text_preprocessor.do_ekphrasis_preprocessing(this_sentence)[0]
                                ready[this_id]['sentence'] = this_sentence
                                ready[this_id]['opinions'] = []
                            if data.tag == 'Opinions':
                                for opinion in data:
                                    opinion_count += 1
                                    ready[this_id]['opinions'].append(
                                        {'target': opinion.get("target"),
                                        'category': opinion.get("category"),
                                        'polarity': opinion.get("polarity"),
                                        'to': int(opinion.get("to")),
                                        'from': int(opinion.get("from")) })
            # sanity check: we got as many opinion tuples as the SemEval paper says
            assert(opinion_count == TEST_OPINIONS or opinion_count == TRAIN_OPINIONS)
            dump_gzip(ready, file_out)

        self.ready_train = load_gzip(self.path_train_ready)
        self.ready_test = load_gzip(self.path_test_ready)
        # sanity check: we got as many sentences as the SemEval paper says
        assert (len(self.ready_train) == TRAIN_SENTENCES and len(self.ready_test) == TEST_SENTENCES)

    def get_test_sentences(self):
        s =  self.get_sentences(self.ready_test)
        assert(len(s) == TEST_SENTENCES)
        return s

    def get_train_sentences(self):
        s = self.get_sentences(self.ready_train)
        assert (len(s) == TRAIN_SENTENCES)
        return s

    def get_sentences(self, data):
        list_dictionaries = data.values()
        sentences = [dictio['sentence'] for dictio in list_dictionaries]
        return np.array(sentences)





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

    embs = load_gzip("Yelp_word_to_emb")

    #x_train = [preprocess(s, return_list=True) for s in x_train]
    x_train = [get_SOW(x, embs) for x in x_train]

    #x_test = [preprocess(s, return_list=True) for s in x_test]
    x_test = [get_SOW(x, embs) for x in x_test]
    return np.squeeze(np.asarray(x_train[:])), y_train, \
           np.squeeze(np.asarray(x_test[:])), y_test

def pad_punctuation_spaces(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

def format_xml_for_NER():

    # train = "data/SemEval2016-Task5-ABSA/SB1/REST/ABSA16_Restaurants_Train_SB1_v2.xml"
    # test = "data/SemEval2016-Task5-ABSA/SB1/REST/EN_REST_SB1_TEST.xml.gold"
    # train_out = "NER-ABSA16_Restaurants_Train_SB1_v2.txt"
    # test_out = "NER-EN_REST_SB1_TEST.xml.gold.txt"

    train = "Aspect-Category-Detection-Model-master/Datasets/ABSA-15_Restaurants_Train_Final.xml"
    test = "Aspect-Category-Detection-Model-master/Datasets/ABSA15_Restaurants_Test.xml"
    train_out = "NER-ABSA-15_Restaurants_Train+Test_Final.txt"
    test_out = "NER-ABSA-15_Restaurants_Test.txt"






## FOR NER
'''
   # always opinion after review text
                            chars = 0
                            review_text = pad_punctuation_spaces(review_text)
                            review_text = review_text.strip()
                            for w in review_text.split(" "):
                                for cat, span, i in zip(categories, target_spans, range(0, len(categories))):
                                    if chars >= span[0] and chars < span[1]:
                                        if chars == span[0]:
                                            myfile.write(smart_str(w + " B-" + cat + "\n"))
                                            break
                                        else:
                                            myfile.write(smart_str(w + " I-" + cat + "\n"))
                                            break
                                    if i == len(categories) - 1:
                                        myfile.write(smart_str(w + " O" + "\n"))

                                chars += len(w)
                                if not re.match('([.,!?()])', w):
                                    chars += 1
                            myfile.write("\n")

'''

if __name__ == '__main__':
    s = SemEvalData()
    r = s.make_vocabulary()
    g = Komn(s.make_vocabulary())
    xt, yt, xte, yte = s.get_data_sow_and_oneHotVector(g)
    t = 3