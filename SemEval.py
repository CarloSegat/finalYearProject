import xml
import numpy as np

from embeddings.Embeddings import Komn
from TextPreprocessor import TextPreprocessor
from utils import dump_gzip, load_gzip, ROOT_DIR, check_argument_is_numpy, pad_array, do_files_exist, \
    assert_is_one_hot_vector

NO_OPINION_TRAIN = 292
NO_OPINION_TEST = 89
TRAIN_SENTENCES = 2000 - NO_OPINION_TRAIN
TEST_SENTENCES = 676 - NO_OPINION_TEST
TRAIN_OPINIONS = 2507
TEST_OPINIONS = 859

ASPECT_CATEGORIES = {'DRINKS#STYLE_OPTIONS': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'LOCATION#GENERAL': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                     'AMBIENCE#GENERAL': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'FOOD#PRICES': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'FOOD#QUALITY': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'RESTAURANT#PRICES': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                     'DRINKS#QUALITY': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'DRINKS#PRICES': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'FOOD#STYLE_OPTIONS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     'RESTAURANT#GENERAL': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                     'SERVICE#GENERAL': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                     'RESTAURANT#MISCELLANEOUS': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
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
        self.dependency_tagged_sentences = ROOT_DIR + '/data/' + 'all_tagged_sentences'
        self.stanford_input_name = "inputStanNLP.txt"
        self.text_preprocessor = TextPreprocessor()

        if do_files_exist(self.path_train_ready, self.path_test_ready):
            self.ready_train = load_gzip(self.path_train_ready)
            self.ready_test = load_gzip(self.path_test_ready)
        else:
            self.load_train_and_test()

        if do_files_exist(self.dependency_tagged_sentences):
            self.ready_tagged = load_gzip(self.dependency_tagged_sentences)
        else:
            self.prepare_tagged_sentences()
            self.ready_tagged = load_gzip(self.dependency_tagged_sentences)

    def get_all_sentences(self):
        train_sentences = self.get_train_sentences()
        test_sentences = self.get_test_sentences()
        return np.concatenate((train_sentences, test_sentences))


    def get_train_x_y_test_x_y(self, preprocessing_options=None):
        #TODO right now a string is returned as train x,
        # change that but add the preprocessing option
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
                x.append(sentence)
                y.append(np.array(aspect_categories))
            train_x_train_y_test_x_test_y.append(np.array(x))
            train_x_train_y_test_x_test_y.append(np.array(y))
        return train_x_train_y_test_x_test_y

    def make_multilabel_1hot_vector(self, aspect_categories):
        multiclass_label = np.zeros(12)
        for ac in aspect_categories:
            multiclass_label = np.add(multiclass_label, ASPECT_CATEGORIES[ac])
        assert_is_one_hot_vector(multiclass_label)
        return multiclass_label

    def get_y_train_and_test_multilabel(self):
        raw = self.get_train_x_y_test_x_y()
        y_train = [self.make_multilabel_1hot_vector(l) for l in raw[1]]
        y_test = [self.make_multilabel_1hot_vector(l) for l in raw[3]]
        return np.array(y_train), np.array(y_test)

    def get_x_sow_and_y_onehot(self, embedding):
        y_train, y_test = self.get_y_train_and_test_multilabel()
        raw = self.get_train_x_y_test_x_y()
        x_train = [embedding.get_SOW(s) for s in raw[0]]
        x_test = [embedding.get_SOW(s) for s in raw[2]]
        return x_train, y_train, x_test, y_test

    def get_x_embs_and_y_onehot(self, embedding, pad=True, pad_size=80):

        def get_embeddings(sentences):
            all_embeddings = []
            for s in sentences:
                embs = embedding.get_word_emb_list(s)
                if len(embs) == 0:
                    # No embeddings found for sentence, ignore it
                    continue
                if pad:
                    padded = pad_array(embs, pad_size)
                all_embeddings.append(padded)
            return np.array(all_embeddings)

        raw = self.get_train_x_y_test_x_y()
        x_train = get_embeddings(raw[0])
        x_test = get_embeddings(raw[2])
        y_train, y_test = self.get_y_train_and_test_multilabel()
        return x_train, y_train, x_test, y_test

    def load_train_and_test(self):
        ''' Creates files of this format:
        {sentence_id: { 'sentence':[sentence], 'opinions':[{}, {}]}

        By default LOWERCASING and DECONTRACTION (I've --> I have) are applied

        '''

        removed = 0
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
                        for data in sentence:
                            if data.tag == "text":
                                this_sentence = data.text
                                this_sentence = \
                                    self.text_preprocessor.do_decontraction(this_sentence)
                                this_sentence = [w.lower() for w in this_sentence]
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
                        if len(ready[this_id]['opinions']) == 0:
                            del ready[this_id] # remove sentence with no opinions
                            removed += 1

            # sanity check: we got as many opinion tuples as the SemEval paper says
            assert(opinion_count == TEST_OPINIONS or opinion_count == TRAIN_OPINIONS)
            dump_gzip(ready, file_out)

        self.ready_train = load_gzip(self.path_train_ready)
        self.ready_test = load_gzip(self.path_test_ready)
        # sanity check: we got as many sentences as the SemEval paper says
        assert (len(self.ready_train) + len(self.ready_test)  ==
                TEST_SENTENCES + TRAIN_SENTENCES)

    def get_test_sentences(self):
        s =  self.get_sentences(self.ready_test)
        assert(len(s) == TEST_SENTENCES)
        return s

    def get_train_sentences(self):
        s = self.get_sentences(self.ready_train)
        assert (len(s) == TRAIN_SENTENCES)
        return s

    def get_sentences(self, data):
        list_dictionaries = list(data.values())
        sentences = []
        for d in list_dictionaries:
            try:
                sentences.append(d['sentence'])
            except Exception:
                pass
        return np.array(sentences)

    def prepare_file_for_Stanford_parser(self):
        ''' Standford dependency parser wants a file where each sentences
        is on new line.
        Remmber to use right argument when calling the dependency parser,
        otherwise full stops are used as delimiters.'''
        sentences = self.get_all_sentences()
        if not isinstance(sentences[0], str):
            if len(sentences[0]) == 1:
                sentences = [sen[0] for sen in sentences]
        f = open(ROOT_DIR + '/data/' + self.stanford_input_name, 'w')
        for s in sentences:
            f.write(s)
            f.write('\n')
        f.close()


    def prepare_tagged_sentences(self, path=ROOT_DIR+'/data/'+'all_dependencies.xml',
                                 dependency_type="enhanced-plus-plus-dependencies"):
        '''Takes as input a xml file containing standford annotated sentences.
        Such file can be obtained by calling the following command:
        java edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,depparse -file <INPUT_FILE>

        Where input file contain all the train and test sentences.
        '''
        opinion_count = 0
        ready = {}
        try:
            docs = xml.etree.ElementTree.parse(path).getroot()
        except Exception:
            self.prepare_file_for_Stanford_parser()
            raise(Exception("You don't have the dependecy file. I made a file called "
                  + self.stanford_input_name + " that you should feed to the standford " +
                "dependency tool to obtain the dependecy xml file"))

        for doc in docs:
            for sentences in doc:
                tagged_sentences = []
                for s in sentences:
                    sentence = []
                    for element in s:
                        if element.tag == 'tokens':
                            for token in element:
                               for thing in token:
                                   if thing.tag == 'word':
                                       sentence.append(thing.text)

                        if element.tag == 'dependencies':
                            if element.attrib['type'] == dependency_type:
                                all_dependencies = {}
                                for dependency in element:
                                    dep_type = dependency.attrib['type']
                                    for thing in dependency:
                                        if thing.tag == 'governor':
                                            governor = thing.text.lower()
                                        if thing.tag == 'dependent':
                                            dependent = thing.text.lower()
                                    all_dependencies.setdefault(governor, [])\
                                        .append(dep_type + "_" + dependent)
                                    all_dependencies.setdefault(dependent, [])\
                                        .append(dep_type + "_inv_" + governor)
                                tagged_sentences.append((sentence, all_dependencies))

                                #tagged_sentences.append(bu ild)
        assert(len(tagged_sentences) == TRAIN_SENTENCES + TEST_SENTENCES)
        dump_gzip(tagged_sentences, ROOT_DIR + '/data/' + 'all_tagged_sentences')

    def split_tagged_sentences_into_train_and_test(self):
        row = load_gzip(ROOT_DIR + '/data/' + 'all_tagged_sentences')
        train = []
        test = []
        for i in range(len(row)):
            if i < TRAIN_SENTENCES:
                train.append(np.array(row[i]))
            else:
                test.append(np.array(row[i]))
        return np.array(train), np.array(test)

    def make_vocabulary(self, sentences):
        all_words = [w for sentence in sentences for w in sentence]
        return set(all_words)

    def make_normal_vocabulary(self):
        ''' List of words used '''
        return self.make_vocabulary(self.get_all_sentences())

    def make_syntactical_vocabulary(self):
        '''Returns stuff like [case_of, det_the ... ] for all words and all
        different syntactical usages found in train + test data'''

        s = load_gzip(ROOT_DIR + '/data/all_tagged_sentences')
        d = []
        for e in s:
            d.append(e[0])
            for ws in list(e[1].values()):
                d.append(ws)
        return self.make_vocabulary(d)

    def get_data_syntax_concatenation_sow(self, komn):
        x_test, x_train = self.get_x_train_test_syntax(komn)
        x_train = [np.array(sum(e)) for e in x_train]
        x_test = [np.array(sum(e)) for e in x_test]
        y_train, y_test = self.get_y_train_and_test_multilabel()
        return x_train, y_train, x_test, y_test

    def get_data_syntax_concatenation(self, komn):
        x_test, x_train = self.get_x_train_test_syntax(komn, pad=True)
        y_train, y_test = self.get_y_train_and_test_multilabel()
        return x_train, y_train, x_test, y_test

    def get_x_train_test_syntax(self, komn, pad=False):
        x_train, x_test = [], []
        for i in range(len(self.ready_tagged)):
            sc = komn.get_syntactic_concatenation(self.ready_tagged[i])
            if pad:
                sc = pad_array(sc, 80)
            if i < TRAIN_SENTENCES:
                x_train.append(sc)
            else:
                x_test.append(sc)
        return np.array(x_test), np.array(x_train)

    def get_syntax_setences_for_NER(self):
        pass



def format_xml_for_NER():

    # train = "data/SemEval2016-Task5-ABSA/SB1/REST/ABSA16_Restaurants_Train_SB1_v2.xml"
    # test = "data/SemEval2016-Task5-ABSA/SB1/REST/EN_REST_SB1_TEST.xml.gold"
    # train_out = "2-NER-ABSA16_Restaurants_Train_SB1_v2.txt"
    # test_out = "2-NER-EN_REST_SB1_TEST.xml.gold.txt"

    train = "Aspect-Category-Detection-Model-master/Datasets/ABSA-15_Restaurants_Train_Final.xml"
    test = "Aspect-Category-Detection-Model-master/Datasets/ABSA15_Restaurants_Test.xml"
    train_out = "2-NER-ABSA-15_Restaurants_Train+Test_Final.txt"
    test_out = "2-NER-ABSA-15_Restaurants_Test.txt"






## FOR 2-NER
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
    #s.prepare_tagged_sentences()
    #s.prepare_file_for_Stanford_parser()
    #s.make_syntactical_vocabulary()
    g = Komn(s.make_normal_vocabulary(), s.make_syntactical_vocabulary())
    s.get_x_sow_and_y_onehot_SYNTAX(g)

    # s.prepare_tagged_sentences()
    c, d, e, f = s.get_train_x_y_test_x_y(None)
    a, b = s.split_tagged_sentences_into_train_and_test()

    s.make_syntactical_vocabulary()
    r = s.make_vocabulary()

    xt, yt, xte, yte = s.get_x_embs_and_y_onehot(g)
    xt, yt, xte, yte = s.get_x_sow_and_y_onehot(g)
    t = 3