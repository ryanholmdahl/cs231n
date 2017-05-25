#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 231N 2016-2017
fc_baseline.py: FC Baseline
Sahil Chopra <schopra8@cs.stanford.edu>
Ryan Holmdahl <ryanlh@stanford.edu>
"""

import tensorflow as tf

from data.dataset_builder import Dataset
from model_builder import ModularModel

params = {}
params['lr'] = 1e-3
params['lr_decay'] = 0.98
params['lr_decay_steps'] = 100
params['im_width'] = 32
params['im_height'] = 32
params['im_channels'] = 1
params['in_conv_layers'] = 0
params['in_conv_filters'] = [10]
params['in_conv_dim'] = [5]
params['in_conv_stride'] = [2]
params['fc_layers'] = 6
params['fc_dim'] = [1024, 1024, 1024, 1024, 1024, 1024]
params['embed_channels'] = 100
params['out_conv_layers'] = 0
params['out_conv_filters'] = [8, 3]
params['out_conv_dim'] = [5, 5]
params['out_conv_stride'] = [1, 1]
params["model_name"] = "fc_baseline"
params["ckpt_path"] = "../ckpt"
params["log_path"] = "../log"
params["n_epochs"] = 1000
params["n_eval_batches"] = 10
params["batch_size"] = 32
params["fc_dropout"] = 0
params["use_transpose"] = False

dataset = Dataset((32, 32))
dataset.read_sets("..\\data\\joint_pairs_32")

m = ModularModel(params)
m.build()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    m.fit(sess, saver, dataset.train_examples, dataset.dev_examples)
    m.restore_from_checkpoint(sess, saver)
    m.demo(sess, dataset.train_examples, dataset.dev_examples, 10)