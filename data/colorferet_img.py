#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 231N 2016-2017
colorfet_img.py: Process images from the NIST colorferet database
Sahil Chopra <schopra8@cs.stanford.edu>
Ryan Holmdahl <ryanlh@stanford.edu>
"""
import glob
import os

# ---------
# CONSTANTS
# ---------
GRAY_FERET_CD1_DIR = './colorferet/colorferet/dvd2/gray_feret_cd1/data/images'
GRAY_FERET_CD2_DIR = './colorferet/colorferet/dvd2/gray_feret_cd2/data/images'
FRONTAL_POSE = 'fa'
PROFILE_RIGHT = 'pr'

def find_img_pairs(dirs, img_1_pose, img_2_pose):
    """
    Examines all files in the given directories, constructing a list of img_pair tuples. Each image pair consists
    of an image of a person taken to capture img_1_pose and another taken to capture img_2_pose.

    :param dirs: List of directories to inspect
    :param img_1_pose: The string suffix utilized to indicate img_1_pose
    :param img_2_pose: The string suffix utilized to indicate img_2 pose
    :return: List of (img_1_pose, img_2_pose) images
    """
    img_pairs = []
    un_matched_imgs = {}

    for img_dir in dirs:
        dir_path = '{}/*.tif.bz2'.format(img_dir)
        for fn in glob.glob(dir_path):
            parse = parse_fn(fn)
            if img_1_pose == parse['POSE'] or img_2_pose == parse['POSE']:
                if parse['ID'] in un_matched_imgs.keys():
                    un_matched_img = un_matched_imgs[parse['ID']]
                    if check_pair(parse, un_matched_img, img_1_pose, img_2_pose):
                        img_pairs.append((un_matched_img['ORIG'], parse['ORIG']))
                        un_matched_imgs.pop(parse['ID'])
                else:
                    un_matched_imgs[parse['ID']] = parse
    return img_pairs


def parse_fn(fn):
    """
    File Name Structure:
    The full-size (512 x 768) images have names such as:
    data/images/00012/00012fb0012_930831.tif.bz2
                      |      |      |
                      |      |      |
                      |      |      \________ FERET pose name
                      |      \_______________ the image capture date
                      \______________________ the subject ID

    :param fn: Filename ex. 00012_930831_fb_a.tif.bz2
    :return: Directory parsing of filename
    """
    base = os.path.basename(fn)
    fn_base = os.path.splitext(os.path.splitext(base)[0])[0]
    segments = fn_base.split('_')
    parse = {
        'ID': segments[0][:5],
        'POSE': segments[0][5:7],
        'DATE': segments[1],
        'ORIG': fn,
    }
    return parse


def check_pair(img1_fn_parse, img2_fn_parse, pose1, pose2):
    """
    Check whether two files constitute a pair.

    :param pose1: Pose1 that is desired
    :param pose2: Pose2 that is desired
    :param img1_fn_parse: Parse of image 1
    :param img2_fn_parse: Parse of image 2
    :return:
    """
    if (
            (img1_fn_parse['POSE'] == pose1 and img2_fn_parse['POSE'] == pose2) or
            (img1_fn_parse['POSE'] == pose2 and img2_fn_parse['POSE'] == pose1)
    ):
        if (
                img1_fn_parse['ID'] == img2_fn_parse['ID'] and img1_fn_parse['DATE'] == img2_fn_parse['DATE']
        ):
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    dirs = [GRAY_FERET_CD1_DIR, GRAY_FERET_CD2_DIR]
    result = find_img_pairs(dirs, FRONTAL_POSE, PROFILE_RIGHT)
    print(len(result))
    print(result[-1])
