import os
import csv
from os import listdir
from os.path import isfile, join, basename, splitext
import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import glob


class Dataset:
    def __init__(self, dims):
        self.train_examples = [[], []]
        self.dev_examples = [[], []]
        self.dims = dims
        self.init = False

    def read_pairs(self, file_path):
        if self.init:
            raise Exception("Dataset already initialized.")
        with open (file_path, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip Header
            for row in enumerate(reader):
                row = row[1]
                emotion = int(row[0])
                usage = row[2]
                img_pixels = row[1].split(' ')
                pixels = np.asarray(img_pixels, dtype=np.int32).reshape(self.dims, self.dims)
                if usage == 'Training' or usage == 'PublicTest':
                    self.train_examples[0].append(pixels)
                    self.train_examples[1].append(emotion)
                elif usage == 'PrivateTest':
                    self.dev_examples[0].append(pixels)
                    self.dev_examples[1].append(emotion)
        self.init = True

    def save_sets(self, out_path):
        if not self.init:
            raise Exception("Dataset not yet initialized.")
        dirnames = [os.path.join(out_path, dirname) for dirname in ["train", "dev"]]
        for dirname in dirnames:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        for dirname, example_list in zip(dirnames, [self.train_examples, self.dev_examples]):
            csv_path = os.path.join(dirname, 'emotions.csv')
            with open(csv_path, 'wb') as emotions_file:
                wr = csv.writer(emotions_file, quoting=csv.QUOTE_ALL)
                wr.writerow(example_list[1])
            for i in range(len(example_list[0])):
                x = example_list[0][i]
                imsave(os.path.join(dirname, "{}.png".format(i)), x, format="png")

if __name__ == '__main__':
    d = Dataset(48)
    d.read_pairs('./fer2013/fer2013.csv')
    print(len(d.train_examples[0]))
