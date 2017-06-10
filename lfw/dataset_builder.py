import os
from os import listdir
from os.path import isfile, join

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave

from utils.face_utils import Cropper
from utils.util import path_leaf


class Dataset:
    def __init__(self, dims, split=None):
        self.train_examples = [[], []]
        self.dev_examples = [[], []]
        self.test_examples = [[], []]
        self.dims = dims
        self.split = split
        self.init = False

    def read_samples(self, data_path):
        if self.init:
            raise Exception("Dataset already initialized.")
        dir_files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
        im_list = []
        for file in dir_files:
            im_in = imread(file, mode="L")
            im_list.append(im_in)
        examples_list = [self.train_examples, self.dev_examples, self.test_examples]
        assignments = np.random.choice([0, 1, 2], size=len(im_list), p=self.split)
        for im, i in zip(im_list, assignments):
            examples_list[i][0].append(self.resize(im))
            examples_list[i][1].append([0])
        for example_list in examples_list:
            example_list[0] = np.array(example_list[0])
            example_list[1] = np.array(example_list[1])
        self.init = True

    def resize(self, im):
        new_im = imresize(im, self.dims)
        new_im = np.reshape(new_im, self.dims)
        return new_im

    def valid_ims(self, x_im, y_im):
        if x_im is None or y_im is None:
            return False
        widths = [x_im.shape[0], y_im.shape[0]]
        if min(widths) * 1.0 / max(widths) < 0.5:
            return False
        return True
