import os
import csv
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave


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

    def read_sets(self, root_path):
        if self.init:
            raise Exception("Dataset already initialized.")
        dirnames = [os.path.join(root_path, dirname) for dirname in ["train", "dev"]]
        for dirname, example_list in zip(dirnames, [self.train_examples, self.dev_examples]):
            child_dirs = os.listdir(dirname)
            print(dirname, child_dirs)
            for child_dir in child_dirs:
                x_im = imread(os.path.join(dirname, child_dir, "x.png"), flatten=True)
                y_im = imread(os.path.join(dirname, child_dir, "y.png"), flatten=True)
                example_list[0].append(np.expand_dims(x_im, 2))
                example_list[1].append(np.expand_dims(y_im, 2))
        for example_list in [self.train_examples, self.dev_examples, self.test_examples]:
            example_list[0] = np.array(example_list[0])
            example_list[1] = np.array(example_list[1])
        self.init = True

if __name__ == '__main__':
    d = Dataset(48)
    d.read_pairs('./fer2013/fer2013.csv')
    d.save_sets('./fer2013_data_splits')
