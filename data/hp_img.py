#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 231N 2016-2017
hp_img.py: Process images from the hp_img dataset (http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html)
Sahil Chopra <schopra8@cs.stanford.edu>
Ryan Holmdahl <ryanlh@stanford.edu>
"""

from collections import defaultdict
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize
from os.path import basename
from os.path import splitext
from os.path import exists
from os import makedirs
import glob
import numpy as np
import re
import random
import math

ORIG_DB_DIR = './HeadPoseImageDatabase'
CROPPED_DB_DIR = './HeadPoseImageDatabase_cropped'
IMG_SIZE = 320
NUM_PEOPLE = 14
NUM_POSES = 2
RANDOM_SEED = 231419

def construct_splits(dir, img_params, split):
    """
    :param dir: directory of image files
    :parm img_params: ((input_tilt, input_pan), (output_tilt, output_pan))
    :param split: tuple of decimals (train, val, test)
    :return: ([(train_input, train_output)], [(val_input, val_output)], [(test_input, test_output)])
    """
    random.seed(RANDOM_SEED)
    train_split, val_split, test_split = split
    fns = get_fns(dir)
    escaped_img_params = escape_img_parms(img_params)
    fn_pairings = []

    # Find input, output image pairs for all people across both poses
    for i in range(1, NUM_PEOPLE+1):
        for j in range(1, NUM_POSES+1):
            fn_pairings.append(find_input_output_fns(i, j, escaped_img_params, fns[i]))

    # Split samples into train, val, and test
    random.shuffle(fn_pairings)
    train_max_idx = int(math.floor(train_split * len(fn_pairings)))
    val_max_idx = int(train_max_idx + math.floor(val_split * len(fn_pairings)))

    train = fn_pairings[:train_max_idx]
    val = fn_pairings[train_max_idx: val_max_idx]
    test = fn_pairings[val_max_idx:]
    split_fn_names = (train, val, test)
    return split_fn_names


def read_dataset(dir):
    """
    Read the dataset directory into memory.
    :param dir: directory of the images
    :return: directory of person number -> img_file_names -> numpy representation of images
    """
    imgs = defaultdict(lambda : defaultdict(str))
    img_paths = ['{}/Person{}/*.jpg'.format(dir, str(i).zfill(2)) for i in range(15)]
    for img_path in img_paths:
        for fn in glob.glob(img_path):
            fn_base = splitext(basename(fn))[0]
            img = (imread(fn)[:, :, :3]).astype(np.float32)
            person_number = get_person_number_from_img_path(img_path)
            imgs[person_number][img_path] = img
    return imgs


def crop_photos(dir, face_crop=False):
    """
    Crop the photos in one of two possible ways:
        1) If face_crop = true, then crop using the dimensions provided in the txt file.
        2) If face_crop = false, then crop using the IMG_SIZE dimensions

    :param dir: directory of the photo crops
    :param face_crop: whether to use the dimensions provided in the txt file to crop
    :return: None
    """
    fn_to_img_metadata = defaultdict()

    metadata_paths = ['{}/Person{}/*.txt'.format(dir, str(i).zfill(2)) for i in range(15)]
    img_paths = ['{}/Person{}/*.jpg'.format(dir, str(i).zfill(2)) for i in range(15)]

    for metadata_path in metadata_paths:
        for fn in glob.glob(metadata_path):
            with open(fn) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            fn_to_img_metadata[splitext(basename(fn))[0]] = content

    for img_path in img_paths:
        for fn in glob.glob(img_path):
            print('Processing {}'.format(fn))
            fn_base = splitext(basename(fn))[0]
            img = (imread(fn)[:, :, :3]).astype(np.float32)
            img_metadata = fn_to_img_metadata[fn_base]

            # Note Metadata Structure:
            # [Corresponding Image File]
            #
            # Face
            # [Face Center X]
            # Face Center Y]
            # [Face Width]
            # [Face Height]
            face_x_center = float(img_metadata[3])
            face_y_center = float(img_metadata[4])
            face_width =  float(img_metadata[5])
            face_height = float(img_metadata[6])

            # Crop photo so as to cut out the background
            if face_crop:
                face_x_left = face_x_center - face_width/2.0
                face_x_right = face_x_center + face_width/2.0
                face_y_bot = face_y_center - face_height/2.0
                face_y_top = face_y_center + face_height/2.0
                cropped_img = img[face_y_bot:face_y_top, face_x_left:face_x_right, :]
            else:
                img = upscale_by_smaller_dim(img, IMG_SIZE)
                height_margin = int((img.shape[0] - IMG_SIZE) / 2)
                if height_margin != 0:
                    cropped_img = img[height_margin: -height_margin, :, :]
                else:
                    cropped_img = img
                width_margin = int((img.shape[1] - IMG_SIZE) / 2)
                if width_margin != 0:
                    cropped_img = cropped_img[:, width_margin: - width_margin, :]
            # Save the Cropped Images
            person_num_idx = img_path.index('Person')
            cropped_img_dir = '{}_cropped/{}'.format(dir, img_path[person_num_idx: person_num_idx+8])
            if not exists(cropped_img_dir):
                makedirs(cropped_img_dir)
            cropped_fn = '{}/{}.jpg'.format(cropped_img_dir, fn_base)
            imsave(cropped_fn, cropped_img)


def upscale_by_smaller_dim(im, img_size):
    """ Upscale the image by the smaller dimension.

    :param im: image in numpy array
    :param img_size: desired dimension size
    :return: Upscale image, if one of the dimensions is less than IMG_SIZE
    """
    min_dim, height_is_min = get_min_dim(im)
    if min_dim < img_size:
        upsample_percentage = (img_size / min_dim)
        if height_is_min:
            width_scaled = int(im.shape[1] * upsample_percentage)
            rescaled_shape = (int(img_size), width_scaled)
        else:
            height_scaled = int(im.shape[0] * upsample_percentage)
            rescaled_shape = (height_scaled, int(img_size))
        im = imresize(im, rescaled_shape)
    return im


def get_min_dim(im):
    """ Get the minimum dimension of the image.

    :param im: image in a numpy array
    :return: Return an int of the minimum dimensions
    """
    # Determine minimum when examining width and height of an image
    # Returns minimm dimension and True (if height is min dim)
    # or False (if width is min dim)
    if im.shape[0] < im.shape[1]:
        return im.shape[0], True
    else:
        return im.shape[1], False

def find_input_output_fns(person_number, pose_number, file_params, file_names):
    input_params, out_params = file_params
    person_number = str(person_number).zfill(2)

    # Get input file names
    input_reg = 'person{}{}\d\d{}{}'.format(person_number, pose_number, input_params[0], input_params[1])
    input_r = re.compile(input_reg)
    input_file_names = filter(input_r.match, file_names)

    # Get output file names
    output_reg = 'person{}{}\d\d{}{}'.format(person_number, pose_number, out_params[0], out_params[1])
    output_r = re.compile(output_reg)
    output_file_names = filter(output_r.match, file_names)

    assert len(input_file_names) == 1
    assert len(output_file_names) == 1

    return (input_file_names[0], output_file_names[0])


def get_person_number_from_img_path(img_path):
    """
    Get the person number referenced by and image path.
    :param img_path: path to image
    :return: the person number indicated by the image path
    """
    person_num_idx = img_path.index('Person')
    person_number = int(img_path[person_num_idx + 6: person_num_idx + 8])
    return person_number

def get_fns(dir):
    """
    Get filenames for imagse in directory.
    :param dir: path to image directory
    :return: person_number -> list of filenames for this person
    """
    fns = defaultdict(list)
    img_paths = ['{}/Person{}/*.jpg'.format(dir, str(i).zfill(2)) for i in range(15)]
    for img_path in img_paths:
        for fn in glob.glob(img_path):
            fn_base = splitext(basename(fn))[0]
            person_number = get_person_number_from_img_path(img_path)
            fns[person_number].append(fn_base)
    return fns

def escape_img_parms(img_params):
    """
    Escape img params and format them appropriately.
    :param img_params: ((input_tilt, input_pan), (output_tilt, output_pan))
    :return: Params in string form and having been escaped for regexes
    """
    input_params, output_params = img_params

    def escape_param(param):
        # Escape a single param value
        if param > 0:
            return re.escape('+{}'.format(param))
        else:
            return re.escape('{}'.format(param))

    def escape_params(params):
        # Escape a tuples of params
        return (escape_param(params[0]), escape_param(params[1]))

    return (escape_params(input_params), escape_params(output_params))


if __name__ == '__main__':
    # input_params = ((re.escape('+15'), re.escape('+15')))
    # output_params = (re.escape('+30'), re.escape('+75'))
    input_params = (15, 15)
    output_params = (30, 75)
    img_params = (input_params, output_params)
    split = (.65, .20, .20)
    data_splits = construct_splits(CROPPED_DB_DIR, img_params, split)
    print data_splits
    print len(data_splits[0]), len(data_splits[1]), len(data_splits[2])
