import tensorflow as tf
import numpy as np
from util import show_image_example
from model_builder import ModularModel
from data.dataset_builder import Dataset

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
    for i in range(10):
        show_image_example(sess, m, dataset.train_examples[0][i], dataset.train_examples[1][i], name='train/fig_{}.png'.format(i))
        show_image_example(sess, m, dataset.dev_examples[0][i], dataset.dev_examples[1][i], name='dev/fig_{}.png'.format(i))

