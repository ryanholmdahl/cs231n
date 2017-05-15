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

ORIG_DB_DIR = './HeadPoseImageDatabase'
IMG_SIZE = 320

def construct_splits():
    return None

def read_dataset():
    imgs = defaultdict(lambda: defaultdict('img_file_name'))

def crop_photos(dir, face_crop=False):
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
    # Upscale image, if one of the dimensions is less than IMG_SIZE
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
    # Determine minimum when examining width and height of an image
    # Returns minimm dimension and True (if height is min dim)
    # or False (if width is min dim)
    if im.shape[0] < im.shape[1]:
        return im.shape[0], True
    else:
        return im.shape[1], False

if __name__ == '__main__':
    crop_photos(ORIG_DB_DIR)