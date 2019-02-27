from keras.utils import Sequence
import itertools
from utils import ROOT_DIR
import numpy as np

class AmazonSequence(Sequence):

    def __init__(self, file_name="data/user_dedup.json",  batch_size=32):
        self.data_iterator = open(ROOT_DIR + file_name)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        data_for_batch = list(itertools.islice(self.data_iterator, self.batch_size)) # [{...},...]
        data_for_batch = [eval(data_point) for data_point in data_for_batch]
        batch_x = [x["reviewText"] for x in data_for_batch]
        batch_y = [x["overall"] for x in data_for_batch]

        return np.array([
            batch_x, np.array(batch_y)

helpful_count = 0
sentences = 0
file_number = 0
with open(ROOT_DIR + "data/user_dedup.json") as infile:
    for line in infile:
        line = eval(line)
        sentences = sentences + 1
        if line["overall"] >= 4.0:
            helpful_count = helpful_count + 1

        if sentences % 100000:
            print("total: " + str(helpful_count))
            exit(0)