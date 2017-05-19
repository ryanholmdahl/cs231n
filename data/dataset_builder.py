from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

from os import listdir
from os.path import isfile, join
from util import path_leaf
import numpy as np
import os
from face_utils import Cropper


class Dataset:
    def __init__(self, dims, split=None):
        self.train_examples = [[], []]
        self.dev_examples = [[], []]
        self.test_examples = [[], []]
        self.dims = dims
        self.split = split
        self.init = False
        self.front_cropper = Cropper(
            'C:\Program Files\OpenCV\opencv\sources\data\lbpcascades\lbpcascade_frontalface.xml')
        self.profile_cropper = Cropper(
            'C:\Program Files\OpenCV\opencv\sources\data\lbpcascades\lbpcascade_profileface.xml', True)

    def read_pairs(self, data_path):
        if self.init:
            raise Exception("Dataset already initialized.")
        dir_files = [join(data_path, f) for f in listdir(data_path) if isfile(join(data_path, f))]
        file_dict = {}
        for file in dir_files:
            im_in = imread(file, mode="L")
            filename = path_leaf(file)
            dataset_type = (filename.split("_")[1]).split(".")[0]
            name = filename.split("_")[0]
            if name not in file_dict:
                file_dict[name] = {}
            file_dict[name][dataset_type] = im_in
        examples_list = [self.train_examples, self.dev_examples, self.test_examples]
        assignments = np.random.choice([0, 1, 2], size=len(file_dict), p=self.split)
        for name, i in zip(file_dict, assignments):
            x_im = self.front_cropper.get_crop(file_dict[name]["x"])
            y_im = self.profile_cropper.get_crop(file_dict[name]["y"])
            if self.valid_ims(x_im, y_im):
                examples_list[i][0].append(self.resize(x_im))
                examples_list[i][1].append(self.resize(y_im))
        for example_list in examples_list:
            example_list[0] = np.array(example_list[0])
            example_list[1] = np.array(example_list[1])
        self.init = True

    def resize(self, im):
        new_im = imresize(im, self.dims)
        print(new_im.shape, im.shape)
        return new_im

    def valid_ims(self, x_im, y_im):
        widths = [x_im.shape[0], y_im.shape[0]]
        if min(widths) * 1.0 / max(widths) < 0.5:
            return False
        return True

    def save_sets(self, out_path):
        if not self.init:
            raise Exception("Dataset not yet initialized.")
        dirnames = [os.path.join(out_path, dirname) for dirname in ["train", "dev", "test"]]
        for dirname in dirnames:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        for dirname, example_list in zip(dirnames, [self.train_examples, self.dev_examples, self.test_examples]):
            for i in range(len(example_list[0])):
                x = example_list[0][i]
                y = example_list[1][i]
                example_path = os.path.join(dirname, str(i))
                os.makedirs(example_path)
                imsave(os.path.join(example_path, "x.png"), x, format="png")
                imsave(os.path.join(example_path, "y.png"), y, format="png")

    def read_sets(self, root_path):
        if self.init:
            raise Exception("Dataset already initialized.")
        dirnames = [os.path.join(root_path, dirname) for dirname in ["train", "dev", "test"]]
        for dirname, example_list in zip(dirnames, [self.train_examples, self.dev_examples, self.test_examples]):
            child_dirs = os.listdir(dirname)
            print(dirname, child_dirs)
            for child_dir in child_dirs:
                x_im = imread(os.path.join(dirname, child_dir, "x.png"), flatten=True)
                y_im = imread(os.path.join(dirname, child_dir, "y.png"), flatten=True)
                example_list[0].append(x_im)
                example_list[1].append(y_im)
        self.init = True
